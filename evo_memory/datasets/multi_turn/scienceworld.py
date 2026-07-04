"""ScienceWorld dataset – AgentBoard env class + evo_mem streaming adapter.

Requires:
    pip install scienceworld==1.2.3

Data path (set via env var or pass data_path=):
    AGENTBOARD_DATA_PATH=/path/to/agentboard/data
    → expects $AGENTBOARD_DATA_PATH/scienceworld/test.jsonl

Each test.jsonl record carries additional_info.{env_name, var}, which maps directly
to scienceworld's env.load(env_name, var) — the authoritative, version-stable link.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ..base import DatasetSplit, MultiTurnDataset, TaskInstance

_AGENTBOARD_DEFAULT = os.environ.get(
    "AGENTBOARD_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agentboard", "data"),
)

# AgentBoard scienceworld simplifications (build_simplification_str): 4 flags WITHOUT
# teleportAction, so navigation difficulty is preserved. Using "easy" would enable all 6
# flags (incl. teleportAction), letting the agent teleport and trivializing navigation —
# diverging from the AgentBoard baseline.
_SIMPLIFICATION_STR = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"


# ─────────────────────────────────────────────────────────────────────────────
# AgentBoard Scienceworld environment class (copied; registry/pdb/
# label_path/from_config removed — subgoals wired in via get_environment)
# ─────────────────────────────────────────────────────────────────────────────

class Scienceworld:
    def __init__(self,
                 serverPath=None,
                 envStepLimit=100,
                 ):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv("", serverPath, envStepLimit=envStepLimit)
        self.reward = 0.
        self.done = False
        # subgoals (regex patterns) are wired in by the dataset's get_environment;
        # finished_sub_goal is the parallel binary completion array
        self.selected_obs = ''
        self.finished_sub_goal = []

    def load(self, task_name, var, simplificationStr):
        return self.env.load(task_name, var, simplificationStr=simplificationStr)

    def inventory(self):
        return self.env.inventory()

    def parseAction(self, action):
        action = action.strip()
        return action

    def step(self, action):
        action = self.parseAction(action)
        observation = ''
        if action == "check valid actions":
            valid_actions = ", ".join(self.get_action_space())
            observation = f"Choose an action from these valid actions: {valid_actions}"
            return observation, self.reward, self.done, {"success": self.done, "progress": self.reward}
        else:
            observation, _, _, info = self.env.step(action)
            if self.selected_obs:
                self._check_temperature_string(observation, self.selected_obs)
            self.reward = self.get_reward()
            self.done = self._check_is_done(self.selected_obs) if self.selected_obs else False
            info = info or {}
            info["success"] = self.done
            info["progress"] = self.reward
            return observation, self.reward, self.done, info

    def get_action_space(self, abstract=True):
        svalid_actions = []
        if abstract:
            for a in self.env.get_possible_actions():
                if "reset" not in a:
                    svalid_actions.append(a)
        else:
            valid_actions = self.env.get_valid_action_object_combinations_with_templates()
            forbidden_words = ["teleport", "connect", "dunk", "eat", "flush", "close door"]
            for valid_action in valid_actions:
                v = valid_action['action']
                for fw in forbidden_words:
                    if fw in v:
                        break
                svalid_actions.append(valid_action['action'])
        if "check valid actions" not in svalid_actions:
            svalid_actions.append("check valid actions")
        return svalid_actions

    def getTaskDescription(self):
        return self.env.get_task_description()

    def getGoalProgressStr(self):
        return self.env.get_goal_progress()

    def getGoldActionSequence(self):
        return self.env.get_gold_action_sequence()

    def reset(self):
        self.reward = 0.
        self.done = False
        if self.selected_obs:
            self.finished_sub_goal = [0 for _ in self.selected_obs]
        # ScienceWorldEnv.reset() returns a Jericho-style (observation, info) tuple where
        # info carries a huge `valid` action list. Drop info and append inventory to the
        # observation, matching AgentBoard's init_obs construction.
        obs, _info = self.env.reset()
        return f"{obs}\n{self.env.inventory()}"

    def _check_temperature_string(self, s, selected_obs):
        for i, pattern in enumerate(selected_obs):
            match = re.search(pattern, s)
            if match:
                self.finished_sub_goal[i] = 1.

    def get_reward(self):
        if not self.finished_sub_goal:
            return 0.
        return sum(self.finished_sub_goal) * 1.0 / len(self.finished_sub_goal)

    def _check_is_done(self, selected_obs):
        return sum(self.finished_sub_goal) >= len(selected_obs)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ScienceWorldDataset(MultiTurnDataset):
    """ScienceWorld dataset backed by AgentBoard test.jsonl + real ScienceWorldEnv.

    90 test instances across 62 unique goals / 30 task types.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        **kwargs,
    ):
        agentboard_root = data_path or _AGENTBOARD_DEFAULT
        self._agentboard_root = agentboard_root
        jsonl_path = os.path.join(agentboard_root, "scienceworld", "test.jsonl")
        super().__init__(data_path=jsonl_path, split=split, **kwargs)

    @property
    def name(self) -> str:
        return "scienceworld"

    def _load_data(self) -> List[TaskInstance]:
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"ScienceWorld test.jsonl not found at {self.data_path!r}. "
                "Set AGENTBOARD_DATA_PATH or pass data_path= to ScienceWorldDataset."
            )

        instances = []
        with open(self.data_path) as f:
            for line in f:
                rec = json.loads(line)
                goal_text = rec["goal"].strip()

                # Map directly via the AgentBoard-recorded (env_name, var) instead of
                # matching goal text against the installed sim. The old goal-text lookup
                # silently dropped 55/90 tasks because scienceworld==1.2.3's descriptions
                # drift from test.jsonl, and the fuzzy [:60] match collided on tasks that
                # share an identical goal string. additional_info is authoritative and
                # version-stable — env.load(env_name, var) reproduces the exact goal text.
                ai = rec.get("additional_info", {})
                task_name = ai.get("env_name")
                var_idx = ai.get("var")
                if task_name is None or var_idx is None:
                    continue

                subgoals = rec["subgoals"]
                if isinstance(subgoals, str):
                    subgoals = [s.strip() for s in subgoals.split("\n") if s.strip()]

                instances.append(TaskInstance(
                    task_id=f"scienceworld_{rec['id']}",
                    input_text=goal_text,
                    target="success",
                    metadata={
                        "id": rec["id"],
                        "task_name": task_name,
                        "var_idx": var_idx,
                        "subgoals": subgoals,
                    },
                    difficulty=rec.get("difficulty"),
                    domain="science",
                ))
        return instances

    def get_environment(self, task_instance: TaskInstance) -> Scienceworld:
        env = Scienceworld(envStepLimit=100)
        env.load(
            task_instance.metadata["task_name"],
            task_instance.metadata["var_idx"],
            simplificationStr=_SIMPLIFICATION_STR,
        )
        # Wire subgoal regex patterns into the env so step() can track progress.
        # Without this, selected_obs stays empty and progress is stuck at 0.0.
        env.selected_obs = task_instance.metadata["subgoals"]
        env.finished_sub_goal = [0 for _ in env.selected_obs]
        return env

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        return (
            "You are an agent in a virtual science lab. Interact using these plain-text "
            "commands ({OBJ} = object, {LOC} = location):\n"
            "\n"
            "Manipulation:\n"
            "  open {OBJ} / close {OBJ}        open or close a container/door\n"
            "  pick up {OBJ} / put down {OBJ}  add to / remove from inventory\n"
            "  move {OBJ} to {OBJ}             transfer an object into a container/location\n"
            "  pour {OBJ} into {OBJ}           pour a substance\n"
            "  dunk {OBJ} into {OBJ}           immerse a container in a liquid\n"
            "  mix {OBJ}                        chemically combine a container's contents\n"
            "\n"
            "Inspection:\n"
            "  look around                     survey the current room\n"
            "  look at {OBJ}                   examine an object\n"
            "  look in {OBJ}                   peek inside a container\n"
            "  read {OBJ}                      read written content\n"
            "\n"
            "Devices:\n"
            "  activate {OBJ} / deactivate {OBJ}   toggle a device (e.g. sink, stove)\n"
            "  use {OBJ} [on {OBJ}]                use a device/item (e.g. thermometer on water)\n"
            "\n"
            "Movement:\n"
            "  go to {LOC}                     move to a connected room\n"
            "\n"
            "Misc:\n"
            "  focus on {OBJ}                  direct attention to a task-relevant object\n"
            "  eat {OBJ} / flush {OBJ} / wait [DURATION]\n"
            "\n"
            "Information:\n"
            "  task                            recap the current objective\n"
            "  inventory                       list items you are carrying\n"
            "\n"
            "Special command: 'check valid actions' lists currently available actions.\n"
            "If your action cannot be understood (e.g. the reply is 'Unknown action.' or "
            "'No known action matches that input.'), the command was not recognized — issue "
            "'check valid actions' to see the available actions, then choose one."
        )

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        success = prediction.lower() == "success"
        return {"success": success, "progress": 1.0 if success else 0.0}
