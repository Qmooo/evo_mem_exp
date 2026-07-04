"""AlfWorld dataset – AgentBoard env class + evo_mem streaming adapter.

Requires:
    pip install alfworld==0.4.2

Game files must be downloaded:
    alfworld-download --data-dir $AGENTBOARD_DATA_PATH/alfworld

Data path (set via env var or pass data_path=):
    AGENTBOARD_DATA_PATH=/path/to/agentboard/data
    → game files at $AGENTBOARD_DATA_PATH/alfworld/json_2.1.1/{valid_seen,valid_unseen,valid_train,train}/
    → JSONL  at     $AGENTBOARD_DATA_PATH/alfworld/test.jsonl
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..base import DatasetSplit, MultiTurnDataset, TaskInstance

_AGENTBOARD_DEFAULT = os.environ.get(
    "AGENTBOARD_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agentboard", "data"),
)

# AgentBoard identifies each test.jsonl task with its exact game via
# additional_info.description (== "<task_folder>/<trial>"); the batch AlfWorld
# class keys labeled_data by it and looks it up at reset() through
# info['extra.gamefile']. We resolve the same game deterministically across all
# json_2.1.1 splits, because AgentBoard's referenced scenes are spread over
# valid_seen / valid_unseen / valid_train / train (not valid_seen alone).
_SPLITS = ("valid_seen", "valid_unseen", "valid_train", "train")


def _resolve_game_file(json_root: str, description: str) -> Optional[str]:
    """Resolve the exact game.tw-pddl for an AgentBoard `description`
    ('<task_folder>/<trial>'), searching every json_2.1.1 split. Returns None
    if no split contains the referenced game."""
    rel = os.path.join(*description.split("/"))
    for split in _SPLITS:
        gf = os.path.join(json_root, split, rel, "game.tw-pddl")
        if os.path.exists(gf):
            return gf
    return None


def _parse_subgoal_patterns(raw: List[str]) -> List[str]:
    """Strip 'Subgoal N: ' prefix from AgentBoard subgoal entries."""
    patterns = []
    for s in raw:
        cleaned = re.sub(r"^Subgoal\s+\d+:\s*", "", s.strip())
        if cleaned:
            patterns.append(cleaned)
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# Per-task environment (one textworld game per task, for EvoMemMultiEnvironment)
# ─────────────────────────────────────────────────────────────────────────────

class AlfWorld:
    """Single textworld game env for per-task EvoMemMultiEnvironment use.

    Loads one game file directly via textworld.gym (batch_size=1) rather than
    scanning directories through a global config, so each task runs an isolated
    environment instance.
    """

    def __init__(self, game_file: str, subgoals: List[str], difficulty: str = "hard") -> None:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self.game_file = game_file
        self.sub_goal = subgoals
        self.difficulty = difficulty
        self.finished_sub_goal: List[float] = []
        self.valid_actions: List[str] = []
        self.init_obs = ""
        self.env_ob = ""
        self.reward = 0.0
        # Terminal signal is `won` (PDDL goal met), NOT the env's `done`: with the
        # step-limit wrapper disabled (max_episode_steps=None), the loop's step
        # budget is owned solely by the ExecutorAgent's max_steps. `done` from
        # textworld would fold in step-limit timeouts, so we never rely on it.
        self.won = False

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=None,   # no env-side step cap; ExecutorAgent.max_steps bounds the loop
            wrappers=[AlfredDemangler, AlfredInfos],
        )
        self._env = textworld.gym.make(env_id)

    def reset(self) -> str:
        ob, info = self._env.reset()
        self.valid_actions = info["admissible_commands"][0]
        lines = "\n".join(ob[0].split("\n\n")[1:]).split("\n")
        self.init_obs = lines[0] if lines else ob[0]
        self.env_ob = self.init_obs
        # N+1 slots: indices 0..N-1 are the N intermediate subgoal regexes
        # (see/pick/cool/heat/clean); the extra slot N stands for TERMINAL task
        # completion, which the subgoal regexes never cover (e.g. "put a cool
        # tomato in microwave" has subgoals see/pick/cool but none for the final
        # placement). That slot is credited only when the env certifies a win,
        # so progress tops out at N/(N+1) until the task is actually solved.
        self.finished_sub_goal = [0.0] * (len(self.sub_goal) + 1)
        self.reward = 0.0
        self.won = False
        return self.init_obs

    def _process_ob(self, ob: str) -> str:
        if ob.startswith("You arrive at loc "):
            ob = ob[ob.find(". ") + 2:]
        return ob

    def _get_reward(self) -> float:
        # progress = filled slots / (N+1). On a certified win the terminal slot
        # (and, implicitly, every intermediate milestone the task required) is
        # complete, so return the full 1.0. The override is gated on self.won
        # (PDDL goal met) — NOT on done, which also fires on step-limit timeout
        # and would otherwise credit an unsolved episode with a perfect score.
        if self.won:
            return 1.0
        denom = len(self.finished_sub_goal)
        return sum(self.finished_sub_goal) / denom if denom else 0.0

    def _check_subgoals(self, obs: str) -> None:
        for i, pattern in enumerate(self.sub_goal):
            if re.search(pattern, obs):
                self.finished_sub_goal[i] = 1.0

    def get_action_space(self) -> List[str]:
        actions = list(self.valid_actions)
        if "look" not in actions:
            actions.append("look")
        if "check valid actions" not in actions:
            actions.append("check valid actions")
        return actions

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if action.endswith("."):
            action = action[:-1]

        # The `done` we return is self.won (task solved). AlfWorld's PDDL games
        # never enter a "lost" state, so won is the only terminal condition; the
        # env's own `done` flag is deliberately ignored (see __init__).
        if action == "check valid actions":
            # Meta-query: does not step the real env, so won is unchanged.
            valid = ", ".join(self.get_action_space())
            obs = f"Choose an action from these valid actions: {valid}"
            return obs, self.reward, self.won, {"success": self.won, "progress": self.reward}

        if action == "look":
            # Step the real env (consumes a turn, refreshes info) but override the
            # observation with the cached current-room description (env_ob):
            # textworld's own `look` after reset or a move returns an empty "you
            # see nothing", losing the room state.
            _, _, _, info = self._env.step([action])
            if info and "admissible_commands" in info:
                self.valid_actions = info["admissible_commands"][0]
            if info and "won" in info:
                self.won = bool(info["won"][0])
            obs = self._process_ob(self.env_ob)
            self._check_subgoals(obs)
            self.reward = self._get_reward()
            return obs, self.reward, self.won, {"success": self.won, "progress": self.reward}

        observation, _, _, info = self._env.step([action])

        if info and "admissible_commands" in info:
            self.valid_actions = info["admissible_commands"][0]
        if info and "won" in info:
            self.won = bool(info["won"][0])

        obs = self._process_ob(observation[0])
        if "go to" in action or "open" in action:
            if "Nothing happens" not in obs:
                self.env_ob = obs

        self._check_subgoals(obs)
        self.reward = self._get_reward()

        return obs, self.reward, self.won, {"success": self.won, "progress": self.reward}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AlfWorldDataset(MultiTurnDataset):
    """AlfWorld dataset backed by AgentBoard test.jsonl + real textworld game files.

    134 test instances across 6 task types (all difficulty=hard per AgentBoard).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        **kwargs,
    ):
        agentboard_root = data_path or _AGENTBOARD_DEFAULT
        self._agentboard_root = agentboard_root
        jsonl_path = os.path.join(agentboard_root, "alfworld", "test.jsonl")
        super().__init__(data_path=jsonl_path, split=split, **kwargs)

    @property
    def name(self) -> str:
        return "alfworld"

    def _load_data(self) -> List[TaskInstance]:
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"AlfWorld test.jsonl not found at {self.data_path!r}. "
                "Set AGENTBOARD_DATA_PATH or pass data_path= to AlfWorldDataset."
            )

        json_root = os.path.join(self._agentboard_root, "alfworld", "json_2.1.1")
        if not os.path.isdir(json_root):
            raise FileNotFoundError(
                f"AlfWorld game files not found at {json_root!r}. "
                "Run: alfworld-download --data-dir $AGENTBOARD_DATA_PATH/alfworld"
            )

        instances = []
        missing: List[str] = []
        with open(self.data_path) as f:
            for line in f:
                rec = json.loads(line)
                goal = rec["goal"]

                # Deterministic AgentBoard mapping: the exact game is named by
                # additional_info.description, NOT re-derived from the goal text.
                description = rec["additional_info"]["description"]
                game_file = _resolve_game_file(json_root, description)
                if game_file is None:
                    missing.append(description)
                    continue

                subgoals_raw = rec["subgoals"]
                if isinstance(subgoals_raw, str):
                    subgoals_raw = [s for s in subgoals_raw.split("\n") if s.strip()]
                subgoal_patterns = _parse_subgoal_patterns(subgoals_raw)

                instances.append(TaskInstance(
                    task_id=f"alfworld_{rec['id']}",
                    input_text=goal,
                    target="success",
                    metadata={
                        "id": rec["id"],
                        "game_file": game_file,
                        "description": description,
                        "subgoals": subgoal_patterns,
                        "difficulty": rec.get("difficulty", "hard"),
                    },
                    difficulty=rec.get("difficulty", "hard"),
                    domain="household",
                ))

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} AlfWorld game(s) referenced by test.jsonl were not found "
                f"under {json_root!r} (e.g. {missing[0]!r}). Download the full game set: "
                "alfworld-download --data-dir $AGENTBOARD_DATA_PATH/alfworld"
            )
        return instances

    def get_environment(self, task_instance: TaskInstance) -> AlfWorld:
        return AlfWorld(
            game_file=task_instance.metadata["game_file"],
            subgoals=task_instance.metadata.get("subgoals", []),
            difficulty=task_instance.metadata.get("difficulty", "hard"),
        )

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        return (
            "You are in a household environment. Issue plain text commands like:\n"
            "  go to [receptacle]\n"
            "  take [object] from [receptacle]\n"
            "  move [object] to [receptacle]        (place a held object; NOTE: 'put ... in/on' is NOT accepted)\n"
            "  open [receptacle] / close [receptacle]\n"
            "  use [object]                          (operate/turn on, e.g. 'use desklamp 1' to light it)\n"
            "  heat [object] with [appliance] / cool [object] with [appliance] / clean [object] with [appliance]\n"
            "  examine [object/receptacle]\n"
            "  look / inventory\n"
            "Special commands: 'look' (re-show current room), 'check valid actions' (list the exact actions valid right now).\n"
            "Observations list objects with numeric IDs (e.g. 'bowl 2'). Include the ID when referring to them."
        )

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        success = prediction.lower() == "success"
        return {"success": success, "progress": 1.0 if success else 0.0}
