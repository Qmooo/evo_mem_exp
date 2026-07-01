"""BabyAI dataset – AgentBoard env class + evo_mem streaming adapter.

Requires:
    pip install minigrid==2.3.0

Data path (set via env var or pass data_path=):
    AGENTBOARD_DATA_PATH=/path/to/agentboard/data
    → expects $AGENTBOARD_DATA_PATH/babyai/test.jsonl
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium
import minigrid
minigrid.register_minigrid_envs()  # registers BabyAI-* environments into gymnasium

from ..base import DatasetSplit, MultiTurnDataset, TaskInstance

_AGENTBOARD_DEFAULT = os.environ.get(
    "AGENTBOARD_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "agentboard", "data"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants from AgentBoard babyai_env.py
# ─────────────────────────────────────────────────────────────────────────────

all_levels = {
    1: "BabyAI-GoToRedBallGrey-v0",
    2: "BabyAI-GoToRedBall-v0",
    3: "BabyAI-GoToRedBallNoDists-v0",
    4: "BabyAI-GoToObjS6-v0",
    5: "BabyAI-GoToLocalS8N7-v0",
    6: "BabyAI-GoToObjMazeS7-v0",
    7: "BabyAI-GoToImpUnlock-v0",
    8: "BabyAI-GoToSeqS5R2-v0",
    9: "BabyAI-GoToRedBlueBall-v0",
    10: "BabyAI-GoToDoor-v0",
    11: "BabyAI-GoToObjDoor-v0",
    12: "BabyAI-Open-v0",
    13: "BabyAI-OpenRedDoor-v0",
    14: "BabyAI-OpenDoorLoc-v0",
    15: "BabyAI-OpenRedBlueDoorsDebug-v0",
    16: "BabyAI-OpenDoorsOrderN4Debug-v0",
    17: "BabyAI-Pickup-v0",
    18: "BabyAI-UnblockPickup-v0",
    19: "BabyAI-PickupLoc-v0",
    20: "BabyAI-PickupDistDebug-v0",
    21: "BabyAI-PickupAbove-v0",
    22: "BabyAI-PutNextLocalS6N4-v0",
    23: "BabyAI-PutNextS7N4Carrying-v0",
    24: "BabyAI-Unlock-v0",
    25: "BabyAI-UnlockLocalDist-v0",
    26: "BabyAI-KeyInBox-v0",
    27: "BabyAI-UnlockPickupDist-v0",
    28: "BabyAI-BlockedUnlockPickup-v0",
    29: "BabyAI-UnlockToUnlock-v0",
    30: "BabyAI-ActionObjDoor-v0",
    31: "BabyAI-FindObjS7-v0",
    32: "BabyAI-KeyCorridorS6R3-v0",
    33: "BabyAI-OneRoomS20-v0",
    34: "BabyAI-MoveTwoAcrossS8N9-v0",
    35: "BabyAI-SynthS5R2-v0",
    36: "BabyAI-SynthLoc-v0",
    37: "BabyAI-SynthSeq-v0",
    38: "BabyAI-MiniBossLevel-v0",
    39: "BabyAI-BossLevel-v0",
    40: "BabyAI-BossLevelNoUnlock-v0",
}

IDX_TO_ACTION = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
ACTION_TO_IDX = {v: k for k, v in IDX_TO_ACTION.items()}

IDX_TO_OBJECT = {
    0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door",
    5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent",
}
OBJECT_TO_IDX = {v: k for k, v in IDX_TO_OBJECT.items()}

STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

DIR_TO_VEC = [
    np.array((1, 0)),   # right
    np.array((0, 1)),   # down
    np.array((-1, 0)),  # left
    np.array((0, -1)),  # up
]


# ─────────────────────────────────────────────────────────────────────────────
# AgentBoard BabyAI environment class (copied; registry/BaseEnvironment/
# matplotlib/gym/render/from_config stripped)
# ─────────────────────────────────────────────────────────────────────────────

class BabyAI:
    def __init__(self,
                 max_episode_steps=50,
                 game_name="BabyAI-GoToRedBall-v0",
                 seed=1234,
                 game_config=None,
                 render_path="temp/babyai_render",
                 need_render=False,
                 obs_to_reward=None,
                 difficulty="easy",
                 ):
        self.max_episode_steps = max_episode_steps
        self.error_message = {}
        self.game_name = game_name
        self.seed = seed
        self.game_config = game_config
        self.env = gymnasium.make(game_name)
        self._inner = self.env.unwrapped  # direct access to minigrid internals
        self.render_path = render_path
        self.need_render = need_render
        self.obs_to_reward = obs_to_reward
        self.store_all_obs_to_reward = obs_to_reward
        self.difficulty = difficulty
        if self.obs_to_reward is not None:
            if isinstance(self.obs_to_reward[0], list):
                self.num_obs_to_reward = len(self.obs_to_reward[0])
            else:
                self.num_obs_to_reward = len(self.obs_to_reward)
        self.reset()

    def _get_info(self):
        return self.infos

    def _get_obs(self):
        return self.states[-1]

    def _get_goal(self):
        return self.goal

    def _get_history(self):
        return self.history

    def _get_action_space(self):
        return list(self.action_space.keys())

    def _is_done(self):
        return self.done

    def match_style(self, obs, pattern):
        pattern = pattern.strip()
        split_token = "**"
        if "**" not in pattern:
            split_token = "*"
        pattern_list = pattern.strip().split(split_token)
        all_obs = obs.split(".")
        for obs_temp in all_obs:
            flag = True
            for p in pattern_list:
                p = p.strip(".")
                if p not in obs_temp:
                    flag = False
            if flag:
                return True
        return False

    def update_reward(self, obs):
        if self.obs_to_reward is None:
            return
        if len(self.obs_to_reward) == 0:
            return
        if isinstance(self.obs_to_reward[0], list):
            need_to_award = False
            path_length = len(self.obs_to_reward[0])
            for i in range(path_length):
                for obs_temp in self.obs_to_reward:
                    if self.match_style(obs, obs_temp[i]):
                        need_to_award = True
                        break
                if need_to_award:
                    self.points += 1
                    self.reward = max(self.reward, self.points / self.num_obs_to_reward)
                    for obs_temp in self.obs_to_reward:
                        obs_temp.remove(obs_temp[i])
                    break
        else:
            for pattern in self.obs_to_reward:
                if self.match_style(obs, pattern):
                    self.points += 1
                    self.reward = max(self.reward, self.points / self.num_obs_to_reward)
                    self.obs_to_reward.remove(pattern)
                    break

    def update(self, action, obs, reward, done, infos):
        for k, v in infos.items():
            self.infos[k] = v
        self.done = done
        self.history.append(("action", action))
        self.history.append(("reward", reward))
        new_obs, new_action_space = self.postprocess_obs(obs)
        if self.done:
            new_obs += "\n The task is completed."
        if self.obs_to_reward is not None:
            self.update_reward(new_obs)
        else:
            self.reward = reward
        # NOTE: previously done was demoted to False when reward <= 0.5, even
        # after the env itself signalled done. That decoupled the "task
        # completed" text (appended above on raw done) from the returned done
        # flag, so the executor never terminated and looped to the step limit.
        # We now trust the env's done signal; reward is still reported as
        # progress. reward == 1 can additionally promote done when the env has
        # not signalled it yet.
        if self.reward == 1 and not self.done:
            self.done = True
            new_obs += "\n The task is completed."
        self.history.append(("state", new_obs))
        self.states.append(new_obs)
        self.action_space = new_action_space
        self.steps += 1
        self.obs_2d = obs["image"]
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
        self.infos["success"] = self.done
        self.infos["progress"] = float(self.reward)

    def update_info(self, action, info):
        self.history.append(("action", action))
        self.history.append(("reward", self.reward))
        self.history.append(("state", info))
        self.states.append(info)
        self.steps += 1
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
        self.infos["success"] = self.done
        self.infos["progress"] = float(self.reward)

    def get_next_pos(self, pos, action, dir):
        if action == 0:
            dir = (dir - 1) % 4
        elif action == 1:
            dir = (dir + 1) % 4
        elif action == 2:
            dir_vec = DIR_TO_VEC[dir]
            pos = tuple(pos + dir_vec)
        return pos, dir

    def find_path(self, init_pos, goal, all_objs, all_barriers, init_dir, xrange, yrange, arrive=False):
        all_things = all_objs + all_barriers
        pos = init_pos
        dir = init_dir
        graph = dict()
        queue = [(pos, dir)]
        state = set()
        while len(queue) > 0:
            pos, dir = queue.pop(0)
            state.add((pos, dir))
            if arrive:
                if pos[0] == goal[0] and pos[1] == goal[1]:
                    path = []
                    while (pos, dir) != (init_pos, init_dir):
                        (pos, dir), action = graph[(pos, dir)]
                        path.append(action)
                    path = path[::-1]
                    return path
            else:
                if goal[0] - pos[0] == DIR_TO_VEC[dir][0] and goal[1] - pos[1] == DIR_TO_VEC[dir][1]:
                    path = []
                    while (pos, dir) != (init_pos, init_dir):
                        (pos, dir), action = graph[(pos, dir)]
                        path.append(action)
                    path = path[::-1]
                    return path
            for action in [2, 0, 1]:
                new_pos, new_dir = self.get_next_pos(pos, action, dir)
                is_obstacle = False
                for obj in all_things:
                    if new_pos[0] not in xrange or new_pos[1] not in yrange:
                        is_obstacle = True
                        break
                    if (new_pos, new_dir) in state:
                        is_obstacle = True
                        break
                    if obj["abs_pos"] == new_pos:
                        if "wall" in obj["name"] or "box" in obj["name"] or "lava" in obj["name"] or "ball" in obj["name"] or "key" in obj["name"]:
                            is_obstacle = True
                            break
                if not is_obstacle:
                    queue.append((new_pos, new_dir))
                    graph[(new_pos, new_dir)] = ((pos, dir), action)
        return None

    def postprocess_obs(self, obs):
        _, vis_mask = self._inner.gen_obs_grid()
        view_size = self._inner.agent_view_size
        pos = self._inner.agent_pos
        f_vec = self._inner.dir_vec
        r_vec = self._inner.right_vec
        top_left = pos + f_vec * (view_size - 1) - r_vec * (view_size // 2)
        vecs = - f_vec + r_vec
        boarders = top_left + view_size * vecs
        xboarder = boarders[0]
        if xboarder < top_left[0]:
            xrange = range(xboarder, top_left[0] + 1)
        else:
            xrange = range(top_left[0], xboarder)
        yboarder = boarders[1]
        if yboarder < top_left[1]:
            yrange = range(yboarder, top_left[1] + 1)
        else:
            yrange = range(top_left[1], yboarder)
        grid = obs["image"]
        dir = obs["direction"]
        all_objs = []
        all_barriers = []
        for vis_j in range(0, view_size):
            for vis_i in range(0, view_size):
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                distance = abs(pos[0] - abs_i) + abs(pos[1] - abs_j)
                if abs_i < 0 or abs_j < 0:
                    continue
                if distance == 0:
                    continue
                obj_type = IDX_TO_OBJECT[grid[vis_i, vis_j, 0]]
                obj_color = IDX_TO_COLOR[grid[vis_i, vis_j, 1]]
                obj_state = IDX_TO_STATE[grid[vis_i, vis_j, 2]]
                if obj_type in ["door", "key", "ball", "box", "goal", "lava", "wall"]:
                    if obj_type == "door":
                        obj_name = obj_color + " " + obj_state + " " + obj_type
                    else:
                        obj_name = obj_color + " " + obj_type
                    all_objs.append({"name": obj_name, "abs_pos": (abs_i, abs_j), "dis": distance})
                if obj_type in ["box", "wall"]:
                    self_dir = DIR_TO_VEC[dir]
                    obj_relative_pos = (abs_i - pos[0], abs_j - pos[1])
                    if np.cross(self_dir, obj_relative_pos) == 0:
                        all_barriers.append({"name": obj_type, "abs_pos": (abs_i, abs_j), "dis": np.dot(self_dir, obj_relative_pos)})
        all_objs.sort(key=lambda x: x["dis"])
        if len(all_objs) > 0:
            cnt_observe = dict()
            obj_description = "In front of you in this room, you can see several objects: "
            for obj_temp in all_objs:
                if 'wall' in obj_temp["name"]:
                    continue
                obj_temp_pos = obj_temp["abs_pos"]
                obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                self_dir = DIR_TO_VEC[dir]
                front_dis = np.dot(self_dir, obj_temp_relative)
                right_dis = np.dot(DIR_TO_VEC[(dir + 1) % 4], obj_temp_relative)
                if right_dis == 0:
                    pos_desc_temp = "right in front of you " + str(int(front_dis)) + " steps away. "
                elif right_dis > 0:
                    pos_desc_temp = str(int(front_dis)) + " steps in front of you and " + str(int(right_dis)) + " steps to your right. "
                else:
                    pos_desc_temp = str(int(front_dis)) + " steps in front of you and " + str(int(-right_dis)) + " steps to your left. "
                if obj_temp["name"] not in cnt_observe:
                    cnt_observe[obj_temp["name"]] = 1
                else:
                    cnt_observe[obj_temp["name"]] += 1
                obj_description += "There is a " + obj_temp["name"] + " " + str(cnt_observe[obj_temp["name"]]) + " " + pos_desc_temp + " "
        else:
            obj_description = "You cannot see any objects within sight."
        barrier_description = "The room has walls around you. "
        if len(all_barriers) > 0:
            all_barriers.sort(key=lambda x: x["dis"])
            barrier_dis_pos = all_barriers[0]["dis"]
            barrier_description += "You are facing a " + all_barriers[0]["name"] + " " + str(barrier_dis_pos) + " steps away. "
        carry_description = ""
        carrying = self._inner.carrying
        if carrying is not None:
            carry_description = "You are carrying a " + carrying.color + " " + carrying.type + "."
        else:
            carry_description = "You are not carrying anything."
        description = obj_description + barrier_description + carry_description
        possible_actions = {"turn left": [0], "turn right": [1]}
        error_message = {}
        if len(all_barriers) == 0 or all_barriers[0]["dis"] > 1:
            possible_actions["move forward"] = [2]
        else:
            error_message["move forward"] = "There is a barrier in front of you, you can't move forward."
        if carrying is None:
            if len(all_objs) > 0:
                cnt_object = dict()
                for i, obj_temp in enumerate(all_objs):
                    if 'wall' in obj_temp["name"]:
                        continue
                    if 'door' in obj_temp["name"]:
                        continue
                    if 'goal' in obj_temp["name"]:
                        continue
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                    self_dir = DIR_TO_VEC[dir]
                    obj_name = obj_temp["name"]
                    front_dis = np.dot(self_dir, obj_temp_relative)
                    right_dis = np.dot(DIR_TO_VEC[(dir + 1) % 4], obj_temp_relative)
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=False)
                    if actions_temp is not None:
                        actions_temp.append(3)
                        if "pickup " + obj_name + " " + str(1) not in possible_actions:
                            cnt_object[obj_name] = 1
                            possible_actions["pickup " + obj_name + " " + str(1)] = actions_temp
                        else:
                            cnt_object[obj_name] += 1
                            possible_actions["pickup " + obj_name + " " + str(cnt_object[obj_name])] = actions_temp
                    else:
                        if "pickup " + obj_name + " " + str(1) not in possible_actions:
                            error_message["pickup " + obj_name + " " + str(1)] = "You cannot pickup " + obj_name + " " + str(1) + ", as there is no path leading to it."
                        else:
                            error_message["pickup " + obj_name + " " + str(cnt_object[obj_name] + 1)] = "You cannot pickup " + obj_name + " " + str(cnt_object[obj_name] + 1) + ", as there is no path leading to it."
        if carrying is not None:
            drop_pos = tuple(pos + DIR_TO_VEC[dir])
            can_drop = True
            for obj_temp in all_objs:
                if obj_temp["abs_pos"] == drop_pos:
                    for obj_type in ["wall", "box", "lava", "ball", "key"]:
                        if obj_type in obj_temp["name"]:
                            can_drop = False
                            break
            if can_drop:
                possible_actions["drop"] = [4]
            else:
                error_message["drop"] = "You cannot drop the object, as there is already an object in front of you."
        else:
            error_message["drop"] = "You cannot drop the object, as you are not carrying anything."
        if len(all_objs) > 0:
            cnt_door = dict()
            for obj_temp in all_objs:
                if 'door' in obj_temp["name"]:
                    if obj_temp["name"] not in cnt_door:
                        cnt_door[obj_temp["name"]] = 1
                    else:
                        cnt_door[obj_temp["name"]] += 1
                if 'open door' in obj_temp["name"]:
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_name = obj_temp["name"]
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=True)
                    if actions_temp is not None:
                        possible_actions["go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = actions_temp
                    else:
                        error_message["go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = "You cannot go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]]) + ", as there is no path leading to it."
                if 'closed door' in obj_temp["name"]:
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_name = obj_temp["name"]
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=False)
                    if actions_temp is not None:
                        possible_actions["toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = actions_temp + [5, 2]
                    else:
                        error_message["toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " " + str(cnt_door) + ", as there is no path leading to it."
                    if actions_temp == []:
                        possible_actions["toggle"] = [5]
                    error_message["go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = "You cannot go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]]) + ", as it is closed. You should toggle it first."
                if 'locked door' in obj_temp["name"]:
                    if carrying is None or carrying.type != 'key':
                        error_message["toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]]) + ", as you are not carrying a key."
                        continue
                    if carrying.color != obj_temp["name"].split(" ")[0]:
                        error_message["toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]]) + ", as the color of the key you are carrying does not match the color of door."
                        continue
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_name = obj_temp["name"]
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=False)
                    if actions_temp is not None:
                        possible_actions["toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = actions_temp + [5, 2]
                    else:
                        error_message["toggle and go through " + obj_temp["name"] + " " + str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " " + str(cnt_door) + ", as there is no path leading to it."
                    if actions_temp == []:
                        possible_actions["toggle"] = [5]
        if len(all_objs) > 0:
            for obj_temp in all_objs:
                if "goal" not in obj_temp["name"]:
                    continue
                obj_temp_pos = obj_temp["abs_pos"]
                actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=True)
                if actions_temp is not None:
                    possible_actions["go to goal"] = actions_temp
                else:
                    error_message["go to goal"] = "You cannot go to the goal, as there is no path leading to it."
        if len(all_objs) > 0:
            cnt_goto = dict()
            for obj_temp in all_objs:
                if "wall" in obj_temp["name"]:
                    continue
                if "goal" in obj_temp["name"]:
                    continue
                obj_name = obj_temp["name"]
                obj_temp_pos = obj_temp["abs_pos"]
                actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=False)
                if actions_temp is not None:
                    if "go to " + obj_name + ' 1' not in possible_actions:
                        possible_actions["go to " + obj_name + ' 1'] = actions_temp
                        cnt_goto[obj_name] = 1
                    else:
                        cnt_goto[obj_name] += 1
                        possible_actions["go to " + obj_name + ' ' + str(cnt_goto[obj_name])] = actions_temp
                else:
                    if "go to " + obj_name + ' 1' not in possible_actions:
                        error_message["go to " + obj_name + ' 1'] = "You cannot go to " + obj_name + ' 1' + ", as there is no path leading to it."
                    else:
                        error_message["go to " + obj_name + ' ' + str(cnt_goto[obj_name] + 1)] = "You cannot go to " + obj_name + ' ' + str(cnt_goto[obj_name] + 1) + ", as there is no path leading to it."
        possible_actions["check available actions"] = []
        self.error_message = error_message
        return description, possible_actions

    def reset(self):
        obs, infos = self.env.reset(seed=self.seed)
        if self.store_all_obs_to_reward is not None:
            self.obs_to_reward = self.store_all_obs_to_reward.copy()
        else:
            self.obs_to_reward = None
        self.goal = self._inner.mission
        if "then" in self.goal:
            self.goal = self.goal.replace("then", "and")
        if "after you" in self.goal:
            self.goal = self.goal.replace("after you", "and")
        description, possible_actions = self.postprocess_obs(obs)
        self.action_space = possible_actions
        self.init_obs = description
        self.infos = infos
        self.states = [self.init_obs]
        self.history = [("state", self.init_obs)]
        self.steps = 0
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
        self.obs_2d = obs["image"]
        self.reward = 0
        self.points = 0
        self.done = False
        return self.init_obs

    def verify_action(self, action, obs):
        if (obs["image"] != self.obs_2d).sum() > 0:
            return True
        else:
            return False

    def check_action_is_valid(self, action):
        if "check" in action:
            return True, None
        if action == "":
            return False, "No change in state."
        if action not in self.action_space:
            if action in self.error_message:
                return False, self.error_message[action]
            else:
                return False, "The action is not recognized. Please check valid actions."
        else:
            return True, None

    def step(self, action):
        action = action.lower()
        action = action.strip()
        is_valid, error = self.check_action_is_valid(action)
        if not is_valid:
            self.update_info(action, error)
            self.infos["action_is_valid"] = False
            return self._get_obs(), self.reward, self.done, self.infos
        elif action == "check available actions" or "check" in action:
            action_info = "You can take the following actions: " + ", ".join(self._get_action_space())
            self.update_info(action, action_info)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done, self.infos
        else:
            action_list = self.action_space[action]
            if action_list == []:
                self.update_info(action, "No change in state.")
                return self._get_obs(), self.reward, self.done, self.infos
            for action_step in action_list:
                obs, reward, done, truncated, infos = self.env.step(action_step)
                if not self.verify_action(action_step, obs):
                    break
                else:
                    self.obs_2d = obs["image"]
            self.update(action, obs, reward, done, infos)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done, self.infos

    def save_log(self, log_path):
        history = self.infos["history"]
        with open(log_path, 'w') as f:
            for item in history:
                item_name = item[0]
                item_content = item[1]
                if item_content is None:
                    continue
                f.write(item_name + ": " + str(item_content) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BabyAIDataset(MultiTurnDataset):
    """BabyAI dataset backed by AgentBoard test.jsonl + real minigrid environments.

    112 test instances across 28 task levels (easy: 86, hard: 26).
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        **kwargs,
    ):
        agentboard_root = data_path or _AGENTBOARD_DEFAULT
        jsonl_path = os.path.join(agentboard_root, "babyai", "test.jsonl")
        super().__init__(data_path=jsonl_path, split=split, **kwargs)

    @property
    def name(self) -> str:
        return "babyai"

    def _load_data(self) -> List[TaskInstance]:
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"BabyAI test.jsonl not found at {self.data_path!r}. "
                "Set AGENTBOARD_DATA_PATH or pass data_path= to BabyAIDataset."
            )
        instances = []
        # AgentBoard generates 4 tasks per level, each with a distinct random
        # mission produced by reset(seed=1234 + within-level index 0..3). The
        # source test.jsonl does not store the seed, so we reconstruct it from a
        # per-subtask running counter (the tasks of one level are consecutive).
        # Without this, get_environment would fall back to seed 1234 for every
        # task, collapsing all 4 missions of a level onto seed 1234's mission and
        # making 3 of every 4 tasks unsolvable (env goal != displayed goal).
        seed_counter: Dict[str, int] = {}
        with open(self.data_path) as f:
            for line in f:
                rec = json.loads(line)
                ai = rec.get("additional_info", {})
                subtask = ai["subtask"]
                within_level_index = seed_counter.get(subtask, 0)
                seed_counter[subtask] = within_level_index + 1
                instances.append(TaskInstance(
                    task_id=f"babyai_{rec['id']}",
                    input_text=rec["goal"],
                    target="success",
                    metadata={
                        "id": rec["id"],
                        "subtask": subtask,
                        "seed": 1234 + within_level_index,
                        "init_obs": ai.get("init_obs", ""),
                        "subgoals": rec["subgoals"] if isinstance(rec["subgoals"], list)
                                    else [s.strip() for s in rec["subgoals"].split("\n") if s.strip()],
                        "difficulty": rec.get("difficulty", "easy"),
                    },
                    difficulty=rec.get("difficulty"),
                    domain="navigation",
                ))
        return instances

    def get_environment(self, task_instance: TaskInstance) -> BabyAI:
        return BabyAI(
            game_name=task_instance.metadata["subtask"],
            seed=task_instance.metadata.get("seed", 1234),
            obs_to_reward=task_instance.metadata.get("subgoals"),
            difficulty=task_instance.metadata.get("difficulty", "easy"),
        )

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        return (
            "You are in a grid-world environment. Use high-level actions like:\n"
            "  turn left / turn right / move forward\n"
            "  pickup [object] [N] / drop\n"
            "  toggle and go through [door] [N] / go through [door] [N]\n"
            "  go to [object] [N] / go to goal\n"
            "  check available actions\n"
            "Object names include color and type (e.g. 'red ball 1')."
        )

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        success = prediction.lower() == "success"
        return {"success": success, "progress": 1.0 if success else 0.0}
