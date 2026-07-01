"""AlfWorld dataset – AgentBoard env class + evo_mem streaming adapter.

Requires:
    pip install alfworld==0.4.2

Game files must be downloaded:
    alfworld-download --data-dir $AGENTBOARD_DATA_PATH/alfworld

Data path (set via env var or pass data_path=):
    AGENTBOARD_DATA_PATH=/path/to/agentboard/data
    → game files at $AGENTBOARD_DATA_PATH/alfworld/json_2.1.1/valid_seen/
    → JSONL  at     $AGENTBOARD_DATA_PATH/alfworld/test.jsonl
"""

from __future__ import annotations

import json
import os
import random
import re
import yaml
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import alfworld
import alfworld.agents.environment
import jsonlines

from ..base import DatasetSplit, MultiTurnDataset, TaskInstance

_AGENTBOARD_DEFAULT = os.environ.get(
    "AGENTBOARD_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agentboard", "data"),
)

# folder prefix → canonical task type key
_FOLDER_TO_TYPE = {
    "pick_and_place_simple":          "simple",
    "look_at_obj_in_light":           "look_at",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep":  "heat",
    "pick_cool_then_place_in_recep":  "cool",
    "pick_two_obj_and_place":         "pick_two",
}


def _goal_to_type_and_obj(goal: str) -> Tuple[str, str]:
    """Parse (type_key, object_lower) from an AlfWorld goal string."""
    g = goal.lower().strip().rstrip(".")

    m = re.match(r"look at (?:an? )?(.+?) under", g)
    if m:
        return "look_at", m.group(1).strip()
    m = re.match(r"examine (?:the |an? )?(.+?) with", g)
    if m:
        return "look_at", m.group(1).strip()

    m = re.match(r"put two (.+?) in ", g)
    if m:
        return "pick_two", m.group(1).strip()

    if "cool" in g:
        m = re.search(r"cool (?:some|an? )?(.+?) (?:and|in)", g)
        if m:
            return "cool", m.group(1).strip()

    if "heat" in g or "hot" in g:
        m = re.search(r"(?:heat (?:the|some|an? )?|put a hot )(.+?) (?:and|in|using|with)", g)
        if m:
            return "heat", m.group(1).strip()

    if "clean" in g:
        m = re.search(r"(?:clean (?:some|the|an? )?|put a clean )(.+?) (?:and|in|with|on)", g)
        if m:
            return "clean", m.group(1).strip()

    m = re.match(r"put (?:some |an? |a )?(.+?) (?:on|in) ", g)
    if m:
        return "simple", m.group(1).strip()

    return "simple", ""


def _parse_subgoal_patterns(raw: List[str]) -> List[str]:
    """Strip 'Subgoal N: ' prefix from AgentBoard subgoal entries."""
    patterns = []
    for s in raw:
        cleaned = re.sub(r"^Subgoal\s+\d+:\s*", "", s.strip())
        if cleaned:
            patterns.append(cleaned)
    return patterns


def _collect_game_files(valid_seen_dir: str) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]]]:
    """Scan valid_seen/ and return {(type_key, object_lower): [game_file_path]} dict."""
    index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    type_only: Dict[str, List[str]] = defaultdict(list)

    for task_folder in sorted(os.listdir(valid_seen_dir)):
        task_path = os.path.join(valid_seen_dir, task_folder)
        if not os.path.isdir(task_path):
            continue
        type_key = None
        obj_lower = ""
        for prefix, key in _FOLDER_TO_TYPE.items():
            if task_folder.startswith(prefix):
                type_key = key
                after_prefix = task_folder[len(prefix) + 1:]
                obj_part = after_prefix.split("-")[0]
                obj_lower = obj_part.lower()
                break
        if type_key is None:
            continue

        for trial in os.listdir(task_path):
            trial_path = os.path.join(task_path, trial)
            game_file = os.path.join(trial_path, "game.tw-pddl")
            if os.path.exists(game_file):
                index[(type_key, obj_lower)].append(game_file)
                type_only[type_key].append(game_file)
                break

    return index, type_only


# ─────────────────────────────────────────────────────────────────────────────
# AgentBoard AlfWorld environment class (copied; registry/pdb stripped)
# ─────────────────────────────────────────────────────────────────────────────

class AlfWorld:
    def __init__(self,
                 split,
                 base_config,
                 batch_size,
                 seed,
                 label_path
                 ):
        with open(base_config) as reader:
            config = yaml.safe_load(reader)
        env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
        env.game_files.sort()
        self.env = env.init_env(batch_size)
        self.valid_actions = []
        self.init_obs = ''
        self.isdone = False
        self.env_ob = self.init_obs
        self.finished_sub_goal = []
        self.labeled_data = {}
        self.sub_goal = []
        with open(label_path, 'r+', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                self.labeled_data[item["additional_info"]['description']] = item
        random.seed(seed)
        self.cur_task_name = ""
        self.reward = 0.

    def reset(self):
        ob, info = self.env.reset()
        self.valid_actions = info["admissible_commands"][0]
        self.init_obs = ('\n'.join(ob[0].split('\n\n')[1:])).split('\n')[0]
        self.goal = ('\n'.join(ob[0].split('\n\n')[1:])).split('\n')[1]
        self.env_ob = self.init_obs
        self.cur_task_name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        self.sub_goal = self.labeled_data[self.cur_task_name]["subgoals"]
        self.difficulty = self.labeled_data[self.cur_task_name]["difficulty"]
        self.finished_sub_goal = [0 for i in range(len(self.sub_goal) + 1)]
        self.reward = 0
        self.isdone = False
        return ob, info

    def step(self, action):
        info = None
        done = self.isdone
        if action.endswith('.'):
            action = action[:-1]
        if action == "look":
            observation, _, done, info = self.env.step([action])
            observation = [self.env_ob]
            done = done[0]
        elif action == "check valid actions":
            valid_actions = ", ".join(self.valid_actions)
            observation = [f"Choose an action from these valid actions: {valid_actions}"]
        else:
            observation, _, done, info = self.env.step([action])
            done = done[0]
        if "go to" in action or "open" in action:
            if "Nothing happens" not in observation[0]:
                self.env_ob = observation[0]
        if info:
            self.valid_actions = info["admissible_commands"][0]
        observation = self._process_ob(observation[0])
        self.isdone = done
        self._check_temperature_string(s=observation, selected_obs=self.sub_goal)
        self.reward = self._get_reward()
        return observation, self.reward, done, info

    def _process_ob(self, ob):
        if ob.startswith('You arrive at loc '):
            ob = ob[ob.find('. ') + 2:]
        return ob

    def _get_reward(self):
        if self.isdone:
            return 1.0
        else:
            return sum(self.finished_sub_goal) * 1.0 / len(self.finished_sub_goal)

    def _check_temperature_string(self, s, selected_obs):
        for i, pattern in enumerate(selected_obs):
            match = re.search(pattern, s)
            if match:
                self.finished_sub_goal[i] = 1.

    def get_action_space(self):
        if "look" not in self.valid_actions:
            self.valid_actions.append("look")
        if "check valid actions" not in self.valid_actions:
            self.valid_actions.append("check valid actions")
        return self.valid_actions


# ─────────────────────────────────────────────────────────────────────────────
# Single-game wrapper (per-task use with EvoMemMultiEnvironment)
# ─────────────────────────────────────────────────────────────────────────────

class SingleGameAlfWorld:
    """Single textworld game env for per-task EvoMemMultiEnvironment use.

    Loads one game file directly via textworld.gym rather than the batch
    AlfWorld class, which requires scanning directories and a global config.
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
        self.isdone = False

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=50,
            wrappers=[AlfredDemangler, AlfredInfos],
        )
        self._env = textworld.gym.make(env_id)

    def reset(self) -> str:
        ob, info = self._env.reset()
        self.valid_actions = info["admissible_commands"][0]
        lines = "\n".join(ob[0].split("\n\n")[1:]).split("\n")
        self.init_obs = lines[0] if lines else ob[0]
        self.env_ob = self.init_obs
        self.finished_sub_goal = [0.0] * len(self.sub_goal)
        self.reward = 0.0
        self.isdone = False
        return self.init_obs

    def _process_ob(self, ob: str) -> str:
        if ob.startswith("You arrive at loc "):
            ob = ob[ob.find(". ") + 2:]
        return ob

    def _get_reward(self) -> float:
        # Progress purely reflects subgoal completion: fraction of subgoal regex
        # patterns matched. No isdone→1.0 override (that hard-coded a full score on
        # the env's done flag regardless of how many subgoals actually matched).
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

        if action == "check valid actions":
            valid = ", ".join(self.get_action_space())
            obs = f"Choose an action from these valid actions: {valid}"
            return obs, self.reward, self.isdone, {"success": self.isdone, "progress": self.reward}

        observation, _, done_list, info = self._env.step([action])
        done = done_list[0]

        if info and "admissible_commands" in info:
            self.valid_actions = info["admissible_commands"][0]

        obs = self._process_ob(observation[0])
        if "go to" in action or "open" in action:
            if "Nothing happens" not in obs:
                self.env_ob = obs

        self._check_subgoals(obs)
        self.isdone = done
        self.reward = self._get_reward()

        return obs, self.reward, done, {"success": done, "progress": self.reward}


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

        valid_seen_dir = os.path.join(
            self._agentboard_root, "alfworld", "json_2.1.1", "valid_seen"
        )
        if not os.path.isdir(valid_seen_dir):
            raise FileNotFoundError(
                f"AlfWorld game files not found at {valid_seen_dir!r}. "
                "Run: alfworld-download --data-dir $AGENTBOARD_DATA_PATH/alfworld"
            )

        obj_index, type_index = _collect_game_files(valid_seen_dir)
        obj_cursors: Dict[Tuple[str, str], int] = defaultdict(int)
        type_cursors: Dict[str, int] = defaultdict(int)

        instances = []
        with open(self.data_path) as f:
            for line in f:
                rec = json.loads(line)
                goal = rec["goal"]
                type_key, obj_lower = _goal_to_type_and_obj(goal)

                key = (type_key, obj_lower)
                if obj_index.get(key):
                    available = obj_index[key]
                    idx = obj_cursors[key] % len(available)
                    obj_cursors[key] += 1
                    game_file = available[idx]
                else:
                    available = type_index.get(type_key, type_index.get("simple", []))
                    idx = type_cursors[type_key] % max(len(available), 1)
                    type_cursors[type_key] += 1
                    game_file = available[idx] if available else None

                if game_file is None:
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
                        "type_key": type_key,
                        "subgoals": subgoal_patterns,
                        "difficulty": rec.get("difficulty", "hard"),
                    },
                    difficulty=rec.get("difficulty", "hard"),
                    domain="household",
                ))
        return instances

    def get_environment(self, task_instance: TaskInstance) -> SingleGameAlfWorld:
        return SingleGameAlfWorld(
            game_file=task_instance.metadata["game_file"],
            subgoals=task_instance.metadata.get("subgoals", []),
            difficulty=task_instance.metadata.get("difficulty", "hard"),
        )

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        return (
            "You are in a household environment. Issue plain text commands like:\n"
            "  go to [location]\n"
            "  take [object] from [location]\n"
            "  put [object] in/on [receptacle]\n"
            "  open/close [object]\n"
            "  heat [object] with [appliance] / clean [object] with [appliance]\n"
            "  look / inventory\n"
            "Special commands: 'look' (re-show current room), 'check valid actions'.\n"
            "Observations list objects with numeric IDs (e.g. 'bowl 2'). Include the ID when referring to them."
        )

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        success = prediction.lower() == "success"
        return {"success": success, "progress": 1.0 if success else 0.0}
