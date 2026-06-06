"""ScienceWorld dataset – AgentBoard env class + evo_mem streaming adapter.

Requires:
    pip install scienceworld==1.2.3

Data path (set via env var or pass data_path=):
    AGENTBOARD_DATA_PATH=/path/to/agentboard/data
    → expects $AGENTBOARD_DATA_PATH/scienceworld/test.jsonl

Goal→(task_name, var_idx) mapping is built lazily and cached to
$AGENTBOARD_DATA_PATH/scienceworld/goal_map.json on first run (~2 min).
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import jsonlines

from ..base import DatasetSplit, MultiTurnDataset, TaskInstance

_AGENTBOARD_DEFAULT = os.environ.get(
    "AGENTBOARD_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agentboard", "data"),
)


def _build_goal_map(data_dir: str) -> Dict[str, Tuple[str, int]]:
    """Build goal_text → (task_name, var_idx) by iterating all ScienceWorld test variations.

    Results are cached to goal_map.json in data_dir. This may take ~2 minutes on first run.
    """
    cache_path = os.path.join(data_dir, "goal_map.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}

    from scienceworld import ScienceWorldEnv

    goal_map: Dict[str, Tuple[str, int]] = {}
    dummy = ScienceWorldEnv("boil", envStepLimit=1)
    task_names = dummy.getTaskNames()

    for task_name in task_names:
        env = ScienceWorldEnv(task_name, envStepLimit=1)
        for var_idx in env.getVariationsTest():
            env.load(task_name, var_idx)
            env.reset()
            goal = env.getTaskDescription().strip()
            if goal not in goal_map:
                goal_map[goal] = (task_name, var_idx)

    with open(cache_path, "w") as f:
        json.dump(goal_map, f, indent=2)
    return goal_map


# ─────────────────────────────────────────────────────────────────────────────
# AgentBoard Scienceworld environment class (copied; registry/pdb/
# label_path loading stripped; from_config removed)
# ─────────────────────────────────────────────────────────────────────────────

class Scienceworld:
    def __init__(self,
                 serverPath=None,
                 envStepLimit=100,
                 label_path=''
                 ):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv("", serverPath, envStepLimit=envStepLimit)
        self.reward = 0.
        self.done = False
        self.label_path = label_path
        self.labels = {}
        self.cur_label = None
        self.modified_goal = ''
        self.selected_obs = ''
        self.finished_sub_goal = []
        if label_path:
            with open(self.label_path, 'r+', encoding='utf-8') as f:
                for item in jsonlines.Reader(f):
                    task_name = item["additional_info"]["env_name"]
                    var = item["additional_info"]["var"]
                    self.labels[f"{task_name}_{var}"] = {
                        "task_name": task_name,
                        "var": var,
                        "modified_goal": item["goal"],
                        "subgoals": item['subgoals'],
                        "difficulty": item["difficulty"],
                    }

    def load(self, task_name, var, simplificationStr):
        env = self.env.load(task_name, var, simplificationStr=simplificationStr)
        if self.labels:
            self.cur_label = self.labels.get(f"{task_name}_{var}")
            if self.cur_label:
                self.selected_obs = self.cur_label["subgoals"]
                self.modified_goal = self.cur_label["modified_goal"]
                self.difficulty = self.cur_label["difficulty"]
                self.finished_sub_goal = [0 for i in range(len(self.selected_obs))]
        return env

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
            for a in self.env.getPossibleActions():
                if "reset" not in a:
                    svalid_actions.append(a)
        else:
            valid_actions = self.env.getValidActionObjectCombinationsWithTemplates()
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
        return self.env.getTaskDescription()

    def getGoalProgressStr(self):
        return self.env.getGoalProgressStr()

    def getGoldActionSequence(self):
        return self.env.getGoldActionSequence()

    def reset(self):
        self.reward = 0.
        self.done = False
        if self.selected_obs:
            self.finished_sub_goal = [0 for _ in self.selected_obs]
        return self.env.reset()

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

        data_dir = os.path.join(self._agentboard_root, "scienceworld")
        goal_map = _build_goal_map(data_dir)

        instances = []
        with open(self.data_path) as f:
            for line in f:
                rec = json.loads(line)
                goal_text = rec["goal"].strip()

                env_info = goal_map.get(goal_text)
                if env_info is None:
                    for key, val in goal_map.items():
                        if goal_text[:60] in key or key[:60] in goal_text:
                            env_info = val
                            break
                if env_info is None:
                    continue

                task_name, var_idx = env_info
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
            simplificationStr="easy",
        )
        return env

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        return (
            "You are in a science lab simulator. Issue plain text commands like:\n"
            "  move to kitchen / look around / pick up [object] / put [object] in [container]\n"
            "  focus on [object] / heat [object] / cool [object] / mix [object] and [object]\n"
            "Special command: 'check valid actions' to list available actions.\n"
            "Type commands one at a time."
        )

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        success = prediction.lower() == "success"
        return {"success": success, "progress": 1.0 if success else 0.0}
