# Packages
import openai
import gymnasium as gym
from Minigrid.minigrid.core.actions import Actions
from Minigrid.minigrid.minigrid_env import MiniGridEnv
import os
from datetime import datetime
import argparse
import wandb
import time
from PIL import Image
import json
import sys
import pyautogui
import numpy as np

# Function to return what GPT returns in sring format
def choose_act(action):
    return str(action)

# Conver the text act into MiniGrid action object, update the inventory as well
def cvt_act(args, inv, act, fro_obj_l):

    act_obj_pair = {"left": Actions.left, "right": Actions.right, "toggle": Actions.toggle,
                        "forward": Actions.forward, "pick up": Actions.pickup, "drop off": Actions.drop}
    
    # Objects: {unseen: 0, empty: 1, wall: 2, floor: 3, door: 4, key: 5, ball: 6, box: 7, goal: 8, lava: 9, agent: 10}
    # Colors: {black: 0, green: 1, blue: 2, purple: 3, yellow: 4, grey: 5}
    # States: {open: 0, closed: 1, locked: 2}
    act_obj = act_obj_pair[act]

    if act == "pick up" and inv == 0 and fro_obj_l[0] in [5, 6, 7]:
        inv = fro_obj_l[0]
    elif act == "drop off" and fro_obj_l[0] in [1, 3]:
        inv = 0

    return inv, act_obj

# Describe the relative location based on the diff_x and diff_y, where the positve values
# means at the front or on the right by default
def describe_location(diff_x, diff_y):
    """Returns a description of a location based on the relative coordinates."""
    description = ""
    # Describe the front/back direction
    if diff_y >= 0:
        description += f"{abs(diff_y)} step(s) in front of you"
    elif diff_y < 0:
        description += f"{abs(diff_y)} step(s) in back of you"
    # Describe the left/right direction
    if diff_x > 0:
        description += f" and {abs(diff_x)} step(s) to your left"
    elif diff_x <= 0:
        description += f" and {abs(diff_x)} step(s) to your right"
    return description


def get_action(args, env_id, world_map, inv, act_his, obs, exp):
    if args.log:
        print(f"\n\n################## Start Deciding ##################\n\n")
    write_log(f"\n\n################## Start Deciding ##################\n\n")
    act_obj_pair = {"0": "left", "1": "right", "2": "toggle",
                    "3": "forward", "4": "pick up", "5": "drop off"}
    if args.input:
        # A demo action when using input
        return "forward"
    else:
        # To get an action, we need first to fill the sys_msg.txt with the args.refresh and use it as system message
        with open(args.sys_temp, 'r', encoding='utf-8') as file:
            sys_msg = file.read()
        sys_msg_s = sys_msg.format(str(args.refresh))
        # Then we need the observation message, which we will fill the act_temp.txt
        with open(args.act_temp, 'r', encoding='utf-8') as file:
            act_msg = file.read()
        obj_map_s = np.array2string(world_map[env_id][0]).replace("'", "").replace("\"","")
        col_map_s = np.array2string(world_map[env_id][1]).replace("'", "").replace("\"","")
        sta_map_s = np.array2string(world_map[env_id][2]).replace("'", "").replace("\"","")
        obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
        if inv == 0:
            inv_s = f"You are not holding anything"
        else:
            inv_s = f"You are holding a {obj_idx[inv]}"
        act_his_s = ", ".join(act_his)
        if args.goal:
            goal = f"Your goal is {obs['mission']}"
            act_msg_s = act_msg.format(obj_map_s, col_map_s, sta_map_s, inv_s, act_his_s, exp, goal)
        else:
            act_msg_s = act_msg.format(obj_map_s, col_map_s, sta_map_s, inv_s, act_his_s, exp, "")
        valid = [i for i in range(6)]
        gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
        if args.goal:
            sys_msg_s += "\nYou will be prompted a goal specific to the environment.\n"
        msg = [{"role": "system", "content": sys_msg_s}]
        fuc_msg = open(args.fuc_msg).read()
        fuc = [{"name": "choose_act","description":fuc_msg,"parameters":{"type":"object", "properties":{"action":{"type":"integer", "description":"the action to take (in integer)","enum":[i for i in range(6)]}}}}]
        msg.append({"role": "user", "content": act_msg_s})
        if args.log:
            print(f"Prompt message = {act_msg_s}")
        write_log(act_msg_s)
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
        while True:
            try:
                rsp = openai.ChatCompletion.create(
                model=gpt_map[args.gpt],
                messages=msg,
                functions = fuc,
                function_call = "auto",
                temperature = args.temp
                )
                rsp_msg = rsp["choices"][0]["message"]
                if rsp_msg.get("function_call"):
                    fuc_l = {
                        "choose_act": choose_act,
                    }
                    fuc_n = rsp_msg["function_call"]["name"]
                    if fuc_n == "choose_act":
                        fuc_c = fuc_l[fuc_n]
                        fuc_args = json.loads(rsp_msg["function_call"]["arguments"])
                        if fuc_args.get("action") in valid:
                            act = fuc_c(
                                action = fuc_args.get("action")
                            )
                break
            except Exception as e:
                if args.log:
                    print(f"Caught an error: {e}\n")
                write_log(f"Caught an error: {e}\n")
                time.sleep(retry_delay)
                retry_delay *= 2  # double the delay each time we retry
    return act_obj_pair[act], act_msg_s

# Get the mapping list between 0,1,2,3 and environment names in a list
def get_env_id_mapping(args):
    file_name = args.env_id_maps
    id_mappings = []
    with open(file_name, "r", encoding = 'utf-8') as file:
        for line in file:
            _, env_name = line.strip().split(", ")
            id_mappings.append(env_name)
    return id_mappings

# Getting the experience based on two observation, action chosen and action history
def get_exp(args, env_id, n_world_map, o_inv, act, o_obs, n_obs, o_world_map, n_inv, act_his):
    if args.log:
        print(f"\n################## Starting Reflection ##################\n")
    write_log(f"\n################## Starting Reflection ##################\n")
    if args.input:
        return "new experience"
    else:
        # To get an experience, we need first to fill the sys_msg.txt with the args.refresh and use it as system message
        with open(args.sys_temp, 'r', encoding='utf-8') as file:
            sys_msg = file.read()
        sys_msg_s = sys_msg.format(str(args.refresh))
        # Then we need the observation message, which we will fill the act_temp.txt
        with open(args.refl_temp, 'r', encoding='utf-8') as file:
            refl_msg = file.read()
        obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
        if o_inv == 0:
            o_inv_s = f"You are not holding anything"
        else:
            o_inv_s = f"You are holding a {obj_idx[inv]}"
        if n_inv == 0:
            n_inv_s = f"You are not holding anything"
        else:
            n_inv_s = f"You are holding a {obj_idx[inv]}"
        if args.goal:
            o_goal = o_obs['mission']
            n_goal = n_obs['mission']
        else:
            o_goal = ""
            n_goal = ""
        act_his_s = ", ".join(act_his) 
        refl_msg_s = refl_msg.format(str(o_world_map[env_id][0]).replace("'", ""), str(o_world_map[env_id][1]).replace("'", ""), str(o_world_map[env_id][2]).replace("'", ""), o_inv_s, act, o_goal, str(n_world_map[env_id][0]).replace("'", ""), str(n_world_map[env_id][1]).replace("'", ""), str(n_world_map[env_id][2]).replace("'", ""), n_inv_s, n_goal, act_his_s, str(args.lim))
        gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
        if args.goal:
            sys_msg_s += "You will be prompted a goal in the environment.\n"
        msg = [{"role": "system", "content": sys_msg_s}]
        if args.log:
            print(f"Prompt Message = \n\n{refl_msg_s}")
        write_log(f"Prompt Message = \n\n{refl_msg_s}")
        msg.append({"role": "user", "content": refl_msg_s})
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
        while True:
            try:
                rsp = openai.ChatCompletion.create(
                    model = gpt_map[args.gpt],
                    messages = msg,
                    temperature = args.temp, 
                    max_tokens = args.lim
                )
                n_exp = rsp["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if args.log:
                    print(f"Caught an error: {e}\n")
                write_log(f"Caught an error: {e}\n")
                time.sleep(retry_delay)
                retry_delay *= 2  # double the delay each time we retry
        if args.log:
            print(f"\n\n***************** Gained Experience *****************\n\n")
            print(f"{n_exp}")
        write_log(f"\n\n***************** Gained Experience *****************\n\n")
        write_log(f"{n_exp}")

        return n_exp

def get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow):
    if arrow == "↑":
        return int(world_map[env_id][pos_x - 1][pos_y])
    elif arrow == "↓":
        return int(world_map[env_id][pos_x + 1][pos_y])
    elif arrow == "←":
        return int(world_map[env_id][pos_x][pos_y - 1])
    elif arrow == "→":
        return int(world_map[env_id][pos_x][pos_y + 1])

# Get the new inventory based on difference between o_obs and n_obs
def get_n_inv(args, n_obs, o_obs):
    indices = np.where(n_obs != o_obs)
    return o_obs[indices[0][0]][indices[1][0]][indices[2][0]]

# Get the saving path for the current argument setting
def get_path(args):
    # Test if the model is getting directions from input
    if args.input:
        dir_n = "INPUT"
    else:
        dir_n = "GPT"
    # Get today's date and format it as MM_DD_YYYY
    timestamp = datetime.today().strftime("%m_%d_%Y"),
    if args.all:
        env_names = "ALL"
    else:
        env_names = "_".join(args.envs)
    arg_list = ["seed", "gpt", "view", "goal", "static", "temp", "steps", "memo", "lim", "refresh"]
    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Combine them to create the full path
    full_path = os.path.join(dir_n, env_names, folder_name, str(timestamp))
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

# Function to get re-spawn position (when seed = 23 only)
def get_pos_m(args):
    # read data from txt file
    with open(args.env_pos, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # parse the data and create matrices
    pos_m = {}
    for line in lines:
        env_id, x, y, arrow = line.strip().split(', ')
        pos_m[int(env_id)] = (int(x), int(y), arrow)

    return pos_m

# Function to return six matrices measuring the exploration ratio (environment for current env_id), 
# they are two:
# A. Environment Ratio
#   1. view:      how much agent has seen versus whole environment
#   2. intr:      how much agent has toggle, pick up, drop off versus all environment
#   3. step:      how much agent has stepped into versus whole environment

# B. Object Ratio
#   1. view:      how much agent has seen versus whole objects list
#   2. intr:      how much agent has toggle, pick up, drop off versus all objects list
def get_ratios(args, env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec):
    env_view_r = np.count_nonzero(env_view_rec) / np.size(env_view_rec) * 100
    env_view_r_s = "{:.3f}%".format(env_view_r)

    env_memo_r = np.count_nonzero(env_memo_rec) / np.size(env_memo_rec) * 100
    env_memo_r_s = "{:.3f}%".format(env_memo_r)

    env_step_r = np.count_nonzero(env_step_rec) / np.size(env_step_rec) * 100
    env_step_r_s = "{:.3f}%".format(env_step_r)

    obj_view_r = np.count_nonzero(obj_view_rec) / np.size(obj_view_rec) * 100
    obj_view_r_s = "{:.3f}%".format(obj_view_r)
    
    obj_intr_r = np.count_nonzero(obj_intr_rec) / np.size(obj_intr_rec) * 100
    obj_intr_r_s = "{:.3f}%".format(obj_intr_r)
    
    return 

# The environment matrix records how many times an agent has seen a portion of object,
# For example, if the agent has seen the object at (1,1), then it will be incremented by 1, with all other places being decreased by 1
# We use this to record and update the world map as agent can remember and forget
# We will set up a memory threshold that how long can an agent remember, with 0 means agent immediately forgets everything once look away
# and 10 means agent remember last 10 action's observations

# The object interact matrix records how many times an agent has interacted with each object, with what interaction
# So it will have a shape of (10, 3), which is recording all three types of interaction for all objects

# The object view matrix records how many times an agent has seen each object
# So it will have a shape of only (10, ) which is recording total number of times agent has seen all objects
def get_rec(args):
    # read data from txt file
    with open(args.env_sizes, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # parse the data and create matrices
    env_view_rec = {}
    env_step_rec = {}
    env_memo_rec = {}
    obj_intr_rec = {}
    obj_view_rec = {}

    for line in lines:
        env_id, h, w = map(int, line.strip().split(','))
        env_view_rec[env_id] = np.zeros((h, w))
        env_step_rec[env_id] = np.zeros((h, w))
        env_memo_rec[env_id] = np.zeros((h, w))
        obj_intr_rec[env_id] = np.array([[0 for i in range(11)] for j in range(3)])
        obj_view_rec[env_id] = np.array([0 for i in range(11)])

    return env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec

# Get the observation map for all environments, with 3-dimension (object, color, status) and height, width.
def get_world_maps(args):
    with open(args.env_sizes, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # parse the data and create world maps
    world_map = {}
    for line in lines:
        env_id, h, w = map(int, line.strip().split(','))
        world_map[env_id] = np.empty((3, h, w), dtype = str)    
        # the three dimensions will be object, color and status, we intiialize them seperately now 
        world_map[env_id][0] = np.full((h, w), "-", dtype = str)
        world_map[env_id][1] = np.full((h, w), "-", dtype = str)
        world_map[env_id][2] = np.full((h, w), "-", dtype = str)

    return world_map

def left_arrow(arrow):
    r_arrow = ""
    if arrow == "↑":
        r_arrow = "←"
    elif arrow == "↓":
        r_arrow = "→"
    elif arrow == "←":
        r_arrow = "↓"
    elif arrow == "→":
        r_arrow = "↑"
    return r_arrow

def right_arrow(arrow):
    r_arrow = ""
    if arrow == "↑":
        r_arrow = "→"
    elif arrow == "↓":
        r_arrow = "←"
    elif arrow == "←":
        r_arrow = "↑"
    elif arrow == "→":
        r_arrow = "↓"
    return r_arrow

def sum_exp(args, n_exp, o_exp, act_his):
    if args.log:
        print(f"\n\n################## Starting Summarizing ##################\n\n")
    write_log(f"\n\n################## Starting Summarizing ##################\n\n")
    if args.input:
        return "summarized experience"
    else:
        # To get an action, we need first to fill the sys_msg.txt with the args.refresh and use it as system message
        with open(args.sys_temp, 'r', encoding='utf-8') as file:
            sys_msg = file.read()
        sys_msg_s = sys_msg.format(str(args.refresh))
        # Then we need the observation message, which we will fill the act_temp.txt
        act_his_s = ", ".join(act_his)
        with open(args.sum_temp, 'r', encoding='utf-8') as file:
            sum_msg = file.read()
        sum_msg_s = sum_msg.format(o_exp, n_exp, act_his_s, str(args.lim))
        gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
        if args.goal:
            sys_msg_s += "You will be prompted a goal in the environment.\n"
        msg = [{"role": "system", "content": sys_msg_s}]
        if args.log:
            print(f"Prompt Message = \n\n{sum_msg_s}")
        write_log(f"Prompt Message = \n\n{sum_msg_s}")
        msg.append({"role": "user", "content": sum_msg_s})
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
        while True:
            try:
                rsp = openai.ChatCompletion.create(
                    model=gpt_map[args.gpt],
                    messages=msg,
                    temperature = args.temp, 
                    max_tokens = args.lim
                )
                c_exp = rsp["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if args.log:
                    print(f"Caught an error: {e}\n")
                write_log(f"Caught an error: {e}\n")
                time.sleep(retry_delay)
                retry_delay *= 2  # double the delay each time we retry
        if args.log:
            print(f"\n\n***************** Summarized Experience *****************\n\n")
            print(f"{c_exp}")
        write_log(f"\n\n***************** Summarized Experience *****************\n\n")
        write_log(f"{c_exp}")
        return c_exp

# Update the position based on the arrow and previous position    
def update_pos(pos_x, pos_y, arrow):
    if arrow == "↑":
        n_pos_x -= 1
    elif arrow == "↓":
        n_pos_x += 1
    elif arrow == "←":
        n_pos_y -= 1
    elif arrow == "→":
        n_pos_y += 1
    return n_pos_x, n_pos_y

# Function to update the records regarding exploration, it needs env_id to determine the environment, 
# act to determine the interaction type, pos to determine the global position and obs to determine
# the target of exploration
def update_rec(args, env_rec, obj_rec, env_id, act, pos, obs, fro_obj_l):
    if obs['direction'] == 0:
        fro_pos = (pos[0], pos[1] + 1)
    elif obs['direction'] == 1:
        fro_pos = (pos[0] + 1, pos[1])
    elif obs['direction'] == 2:
        fro_pos = (pos[0], pos[1] - 1)
    elif obs['direction'] == 3:
        fro_pos = (pos[0] - 1, pos[1])
    n_env_rec = env_rec.copy()
    n_obj_rec = obj_rec.copy()
    if act == "toggle" or act == "pick up" or act == "drop off":
        n_env_rec[env_id][1][fro_pos] = 1

    # Update the obj interact record
    if act == "toggle" or act == "pick up" or act == "drop off":
        n_obj_rec[env_id][1][fro_obj_l[0]] = 1

    # 3. Update the agent's position if action is forward and front is empty space, opened door
    if act == "forward":
        # Update position if the front is door and the status is opened
        if fro_obj_l[0] == 4 and fro_obj_l[2] == 0:
            n_pos = fro_pos
        # Update position if the front is empty or front is goal
        elif fro_obj_l[0] == 1 or fro_obj_l[0] == 8:
            n_pos = fro_pos
        # Else use old position
        else:
            n_pos = pos
    else:
        n_pos = pos
    return n_pos, n_env_rec, n_obj_rec

# 1. We need to update the world map in object level, which we should initialize using pos_x, pos_y, arrow, obs, world_map, env_memo_rec
def update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec):
    # With the new obs, we should first update the env_memo_rec, as it will determine which parts of world map will show
    if arrow == "→":
        # It means the agent is facing right, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        image = obs['image'].transpose(1,0,2)
        rotated_image_obj = np.rot90(image[:, :, 0], k = -1)
        rotated_image_col = np.rot90(image[:, :, 1], k = -1)
        rotated_image_sta = np.rot90(image[:, :, 2], k = -1)
        # 0. Deduct the env_memo_rec by 1 unless 0
        env_memo_rec[env_id] = np.where(env_memo_rec[env_id] > 0, env_memo_rec[env_id] - 1, env_memo_rec[env_id])
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(max(0, pos_x - args.view // 2), min(env_view_rec[env_id].shape[0], pos_x + args.view // 2 + 1)):
            for col in range(pos_y, min(env_view_rec[env_id].shape[1], pos_y + args.view)):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - (pos_x - args.view // 2)][col - pos_y]
                col_name = rotated_image_col[row - (pos_x - args.view // 2)][col - pos_y]
                sta_name = rotated_image_sta[row - (pos_x - args.view // 2)][col - pos_y]
                if obj_name != 0:
                    env_memo_rec[env_id][row][col] = args.memo
                    env_view_rec[env_id][row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[env_id][0][row][col] = str(obj_name)
                    world_map[env_id][1][row][col] = str(col_name)
                    world_map[env_id][2][row][col] = str(sta_name)
                obj_view_rec[env_id][obj_name] += 1
                if row == pos_x and col == pos_y:
                    p_obj, p_col, p_sta = world_map[env_id][0][row][col], world_map[env_id][1][row][col], world_map[env_id][2][row][col]
        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    elif arrow == "↑":
        # It means the agent is facing north, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        image = obs['image'].transpose(1,0,2)
        rotated_image_obj = image[:, :, 0]
        rotated_image_col = image[:, :, 1]
        rotated_image_sta = image[:, :, 2]
        # 0. Deduct the env_memo_rec by 1 unless 0
        env_memo_rec[env_id] = np.where(env_memo_rec[env_id] > 0, env_memo_rec[env_id] - 1, env_memo_rec[env_id])
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(max(0, pos_x - args.view + 1), pos_x + 1):
            for col in range(max(0, pos_y - args.view // 2), min(env_view_rec[env_id].shape[1], pos_y + args.view // 2 + 1)):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - (pos_x - args.view + 1)][col - (pos_y - args.view // 2)]
                col_name = rotated_image_col[row - (pos_x - args.view + 1)][col - (pos_y - args.view // 2)]
                sta_name = rotated_image_sta[row - (pos_x - args.view + 1)][col - (pos_y - args.view // 2)]
                if obj_name != 0:
                    env_memo_rec[env_id][row][col] = args.memo
                    env_view_rec[env_id][row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[env_id][0][row][col] = str(obj_name)
                    world_map[env_id][1][row][col] = str(col_name)
                    world_map[env_id][2][row][col] = str(sta_name)
                obj_view_rec[env_id][obj_name] += 1
                if row == pos_x and col == pos_y:
                    p_obj, p_col, p_sta = world_map[env_id][0][row][col], world_map[env_id][1][row][col], world_map[env_id][2][row][col]
        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    elif arrow == "↓":
        # It means the agent is facing south, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        image = obs['image'].transpose(1,0,2)
        rotated_image_obj = np.rot90(image[:, :, 0], k = -2)
        rotated_image_col = np.rot90(image[:, :, 1], k = -2)
        rotated_image_sta = np.rot90(image[:, :, 2], k = -2)
        # 0. Deduct the env_memo_rec by 1 unless 0
        env_memo_rec[env_id] = np.where(env_memo_rec[env_id] > 0, env_memo_rec[env_id] - 1, env_memo_rec[env_id])
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(pos_x, min(env_view_rec[env_id].shape[0], pos_x + args.view)):
            for col in range(max(0, pos_y - args.view // 2), min(env_view_rec[env_id].shape[1], pos_y + args.view // 2 + 1)):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - pos_x][col - (pos_y - args.view // 2)]
                col_name = rotated_image_col[row - pos_x][col - (pos_y - args.view // 2)]
                sta_name = rotated_image_sta[row - pos_x][col - (pos_y - args.view // 2)]
                if obj_name != 0:
                    env_memo_rec[env_id][row][col] = args.memo
                    env_view_rec[env_id][row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[env_id][0][row][col] = str(obj_name)
                    world_map[env_id][1][row][col] = str(col_name)
                    world_map[env_id][2][row][col] = str(sta_name)
                obj_view_rec[env_id][obj_name] += 1
                if row == pos_x and col == pos_y:
                    p_obj, p_col, p_sta = world_map[env_id][0][row][col], world_map[env_id][1][row][col], world_map[env_id][2][row][col]
        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    elif arrow == "←":
        # It means the agent is facing right, so we update the env_memo_rec accordingly, specifically we 
        # set args.memo to the corresponding observed area while all other values are deducted by 1 unless equals to 0
        # In addition we increment one to the env_view_rec which is basically recording how many times agent has seen that area
        image = obs['image'].transpose(1,0,2)
        rotated_image_obj = np.rot90(image[:, :, 0], k = 1)
        rotated_image_col = np.rot90(image[:, :, 1], k = 1)
        rotated_image_sta = np.rot90(image[:, :, 2], k = 1)
        # 0. Deduct the env_memo_rec by 1 unless 0
        env_memo_rec[env_id] = np.where(env_memo_rec[env_id] > 0, env_memo_rec[env_id] - 1, env_memo_rec[env_id])
        # 1. Update the env_view_rec, env_memo_rec, obj_view_rec, world_map in three channels
        for row in range(max(0, pos_x - args.view // 2), min(env_view_rec[env_id].shape[0], pos_x + args.view // 2 + 1)):
            for col in range(max(0, pos_y - args.view), pos_y + 1):
                # If the object is not unseen, we record it to the env_memo, env_view, obj_view, world_map
                obj_name = rotated_image_obj[row - (pos_x - args.view // 2)][col - (pos_y - args.view + 1)]
                col_name = rotated_image_col[row - (pos_x - args.view // 2)][col - (pos_y - args.view + 1)]
                sta_name = rotated_image_sta[row - (pos_x - args.view // 2)][col - (pos_y - args.view + 1)]
                if obj_name != 0:
                    env_memo_rec[env_id][row][col] = args.memo
                    env_view_rec[env_id][row][col] += 1
                    # Update the world_map object level to be what's seen, except in agent's position we set it to be the arrow
                    world_map[env_id][0][row][col] = str(obj_name)
                    world_map[env_id][1][row][col] = str(col_name)
                    world_map[env_id][2][row][col] = str(sta_name)
                obj_view_rec[env_id][obj_name] += 1
                if row == pos_x and col == pos_y:
                    p_obj, p_col, p_sta = world_map[env_id][0][row][col], world_map[env_id][1][row][col], world_map[env_id][2][row][col]
        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    for row in range(env_memo_rec[env_id].shape[0]):
        for col in range(env_memo_rec[env_id].shape[1]):
            if env_memo_rec[env_id][row][col] == 0:
                world_map[env_id][0][row][col] = "-"
                world_map[env_id][1][row][col] = "-"
                world_map[env_id][2][row][col] = "-"
    return p_obj, p_col, p_sta
# Function to write into the act_temp.txt to hint an action
def write_act_temp(args, world_map, pos, obs, inv, env_id, act_his, c_exp, fro_obj_l):
    dir = obs['direction']
    dir_dic = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}
    dir_s = dir_dic[dir]

    img = obs['image'].transpose(1,0,2)
    act_his_s = ", ".join(act_his)
    with open(args.act_temp, 'r', encoding='utf-8') as file:
        temp = file.read()

    obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
    col_idx = {0: "black", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
    sts_idx = {0: "open", 1: "closed", 2: "locked"}

    fro_obj_s = f"Your front object is {obj_idx[fro_obj_l[0]]} {col_idx[fro_obj_l[1]]} {sts_idx[fro_obj_l[2]]}"

    act_temp_s = temp.format(str(world_map), str(pos), dir_s, str(img), str(inv), str(env_id), act_his_s, c_exp, fro_obj_s)
    
    return act_temp_s

# Function to write the logging infos in to log save file
def write_log(text):
    save_path = get_path(args)
    # Open the file in append mode
    with open(os.path.join(save_path, f"log.txt"), "a", encoding='utf-8') as file:
        # Write the strings to the file
        file.write(text)

# Function to write into the refl_temp.txt to hint an reflection based on past & new observation
def write_refl_temp(args, o_world_map, o_pos, o_obs, o_inv, o_env_id, 
                   act, o_fro_obj_s, n_world_map, n_pos, n_obs, n_inv, 
                   n_env_id, n_fro_obj_s, n_act_his):
    
    dir_dic = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}

    o_img = o_obs['image'].transpose(1,0,2)
    o_dir = o_obs['direction']
    o_dir_s = dir_dic[o_dir]

    n_img = n_obs['image'].transpose(1,0,2)
    n_dir = n_obs['direction']
    n_dir_s = dir_dic[n_dir]

    n_act_his_s = ", ".join(n_act_his)
    
    with open(args.refl_temp, 'r', encoding='utf-8') as file:
        temp = file.read()

    refl_temp_s = temp.format(str(o_world_map), str(o_pos), o_dir_s, str(o_img), str(o_inv), 
                             str(o_env_id), act, o_fro_obj_s, str(n_world_map), str(n_pos), 
                             n_dir_s, str(n_img), str(n_inv), str(n_env_id), n_fro_obj_s, 
                             n_act_his_s, str(args.lim))
    
    return refl_temp_s

# Function to write into the rpt_temp.txt to report the current progress.
def write_rpt_temp(args, env_view, env_intr, env_step, obj_view, obj_intr, pos, 
                   env_view_r_s, env_intr_r_s, env_step_r_s, obj_view_r_s, obj_intr_r_s):
    
    with open(args.rpt_temp, 'r', encoding='utf-8') as file:
        temp = file.read()

    rpt_temp_s = temp.format(str(env_view), str(env_intr), str(env_step), 
                             str(obj_view), str(obj_intr), str(pos),
                             env_view_r_s, env_intr_r_s, env_step_r_s,
                             obj_view_r_s, obj_intr_r_s)
    
    return rpt_temp_s

# Function to write into the sum_temp.txt to hint an summarized experience based past & new experience
def write_sum_temp(args, o_exp, n_exp, act_his):

    with open(args.sum_temp, 'r', encoding='utf-8') as file:
        temp = file.read()
        
    act_his_s = ", ".join(act_his)

    refl_temp_s = temp.format(o_exp, n_exp, act_his_s, str(args.lim))
    
    return refl_temp_s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action = "store_true",
        help = "to load all the environments if given",
    )
    parser.add_argument(
        "--act-temp",
        default = "./utilities/act_msg.txt",
        type = str,
        help = "the location to load your action prompt message template"
    )
    parser.add_argument(
        "--API-key",
        default = "./utilities/API/API_1",
        type = str,
        help = "the location to load your OpenAI API Key"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "list of environment names, see the ./utilities/envs_mapping.txt for mapping between index and env",
        default = ["0"]
    )
    parser.add_argument(
        "--env-id-maps",
        type = str,
        help = "the environment ID and environment name mapping",
        default = "./utilities/env_id_maps.txt"
    )
    parser.add_argument(
        "--env-sizes",
        type = str,
        help = "the environment ID and environment map size",
        default = "./utilities/env_sizes.txt"
    )
    parser.add_argument(
        "--env-pos",
        type = str,
        help = "the re-spawn position of the agent in each map (when seed = 23)",
        default = "./utilities/env_pos_23.txt"
    )
    parser.add_argument(
        "--exp-src",
        type = str,
        help = "the starting experience read path"
    )
    parser.add_argument(
        "--fuc-msg",
        type = str,
        default = "./utilities/fuc_msg.txt",
        help = "the path to read the function prompt message"
    )
    parser.add_argument(
        "--goal",
        action = "store_true",
        help = "whether include the text goal into the observation description"
    )
    parser.add_argument(
        "--gpt",
        type = str,
        choices = ["3", "4"],
        help = "the version of gpt, type version number like 3 or 4",
        default = "3"
    )
    parser.add_argument(
        "--input",
        action = "store_true",
        help = "if the action and experience will be given by user instead of generating from GPT"
    )
    parser.add_argument(
        "--lim",
        type = int,
        default = 1000,
        help = "the words limit to the experience"
    )
    parser.add_argument(
        "--log",
        action = "store_true",
        help = "print the logging informations by print()"
    )
    parser.add_argument(
        "--memo",
        type = int,
        help = "how long can agent remember past scenes",
        default = 5
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "LLM As Agent"
    )
    parser.add_argument(
        "--refl-temp",
        default = "./utilities/refl_msg.txt",
        type = str,
        help = "the location to load your reflect prompt message template"
    )
    parser.add_argument(
        "--refresh",
        type = int,
        default = 20,
        help = "the maximum number of action history"
    )
    parser.add_argument(
        "--rpt-temp",
        default = "./utilities/rpt_msg.txt",
        type = str,
        help = "the location to load your exploration report template format"
    )
    parser.add_argument(
        "--rty-dly",
        type = int,
        default = 5,
        help = "the number of seconds to delay when in OpenAI API Calling"
    )
    parser.add_argument(
        "--screen",
        type = int,
        default = 640,
        help = "set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--seed",
        type = int,
        help = "random seed for reproducing results",
        default = 23
    )
    parser.add_argument(
        "--static",
        action = "store_true",
        help = "the agent will not update experience during the exploration"
    )
    parser.add_argument(
        "--steps",
        type = int,
        default = 2000,
        help = "the maximum numbers of steps each environment will be taken"
    )
    parser.add_argument(
        "--sum-temp",
        default = "./utilities/sum_msg.txt",
        type = str,
        help = "the location to load your summarization prompt message template"
    )
    parser.add_argument(
        "--sys-temp",
        type = str,
        default = "./utilities/sys_msg.txt",
        help = "the location to load your hint message as agent's system background"
    )
    parser.add_argument(
        "--temp",
        type = float,
        default = 0.7,
        help = "the temprature used by the OpenAI API"
    )
    parser.add_argument(
        "--view",
        type = int,
        default = 7,
        help = "set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--wandb",
        action = "store_true",
        help = "whether to use wandb to record experiments"
    )
    args = parser.parse_args()
    save_path = get_path(args)

    # Start running the specified environment(s), for each one, it has limited steps, whether it uses an existing
    # experience or an evolving experience will be dependent on the arguments. 
    if args.log:
        print(f"################## Starting Experiment ##################\n")
        print(f"Configurations are:\n{args}")
    write_log(f"################## Starting Experiment ##################\n")
    write_log(f"Configurations are:\n{args}")

    # Load the experience if it's given, and determine the training process based on --static
    if args.exp_src is not None:
        exp = open(args.exp_src).read()
    else:
    # Initial Experience
        exp = ""

    # Load the API key
    openai.api_key = open(args.API_key).read()

    if args.wandb:
        wandb.init(
            project = args.prj_name,
            name = datetime.now().strftime("Run %Y-%m-%d %H:%M:%S"),
            config = vars(args)
        )
    

    # get the environment list based on --all or --envs, if --all then replace args.envs as all environments
    if args.all:
        args.envs = [str(i) for i in range(61)]
    
    with open(args.sys_temp, 'r', encoding='utf-8') as file:
        sys_msg = file.read()
    sys_msg_s = sys_msg.format(str(args.refresh))
    if args.log:
        print(f"\n################## System Message ##################\n")
        print(sys_msg_s)
    write_log(f"\n################## System Message ##################\n")
    write_log(sys_msg_s)

    # Get the observation map for all environments, with 3-dimension (object, color, status) and height, width.
    world_map = get_world_maps(args)
    # Get the two record matrix for all environments, with environment and object level
    env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec = get_rec(args)
    # The environment ID and enviornment name mapping list
    envs_id_mapping = get_env_id_mapping(args)
    # Get the position mapping for all environments, which include the x, y (in integer) and the direction → string
    pos_m = get_pos_m(args)
    # Iterate over all the environment
    for i in args.envs:
        # The i is a string representing environment ID, e.g. "1"
        env_id = int(i)
        # For each new environment, the inventory is always 0
        inv = 0
        # For every new environment, the action history is always 0 (empty)
        act_his = []
        # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a →
        pos_x, pos_y, arrow = pos_m[env_id]
        # Get environment name from the mapping
        env_name = envs_id_mapping[env_id]
        if args.log:
            print(f"Loading environment = {env_name}")
        write_log(f"Loading environment = {env_name}")

        env: MiniGridEnv = gym.make(
            id = env_name,
            render_mode = "rgb_array",
            agent_view_size = args.view,
            screen_size = args.screen
        )

        if args.wandb:
            scn_table = wandb.Table(columns = ["img", "text", "act", "n_exp", "c_exp"])
            env_table = wandb.Table(columns = ["env_view_rec", "env_step_rec", "env_memo_rec"])
            obj_table = wandb.Table(columns = ["obj_intr_rec", "obj_view_rec"])
            world_map_table = wandb.Table(columns = ["world_map_obj", "world_map_col", "world_map_sta"])

        # Initilize the environment
        obs, state = env.reset(seed=args.seed)
         # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
        p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        # Iterate the agent exploration within the limit of args.steps
        if args.wandb:
            scn_table.add_data(wandb.Image(img), act_msg_s, act, n_exp, c_exp)
            env_table.add_data(str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", ""))
            obj_table.add_data(str(obj_intr_rec[env_id]), str(obj_view_rec[env_id]))
            world_map_table.add_data(str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", ""))
        
        for j in range(args.steps):
            # We get a new action, during which update the record tables
            act, act_msg_s = get_action(args, env_id, world_map, inv, act_his, obs, exp)
            # We get the action from the act_hint, the act is a string format like "pick up"
            # With the new act, we convert it into the actions object
            img_array = env.render()
            img = Image.fromarray(img_array)
            img.save(os.path.join(save_path, f"env_{i}_action_{str(j)}_{act}.png"))
            
            if act == "left":

                # We update the world map, env view, env memo, obj_view, act history, arrow
                arrow = left_arrow(arrow)
                o_world_map = world_map.copy()
                n_obs, reward, terminated, truncated, _ = env.step(Actions.left)
                act_his.append(act)
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his)
                c_exp = sum_exp(args, n_exp, exp, act_his)

                exp = c_exp
                obs = n_obs

            elif act == "right":

                arrow = right_arrow(arrow)
                o_world_map = world_map.copy()
                n_obs, reward, terminated, truncated, _ = env.step(Actions.right)
                act_his.append(act)
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his)
                c_exp = sum_exp(args, n_exp, exp, act_his)

                exp = c_exp
                obs = n_obs

            elif act == "forward":

                o_world_map = world_map.copy()
                n_obs, reward, terminated, truncated, _ = env.step(Actions.forward)
                act_his.append(act)
                if terminated:
                    pos_x, pos_y, arrow = pos_m[env_id]
                    sys.exit()
                else:
                    if not np.array_equal(n_obs, obs):
                        n_pos_x, n_pos_y = update_pos(pos_x, pos_y, arrow)
                world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his)
                c_exp = sum_exp(args, n_exp, exp, act_his)

                exp = c_exp
                obs = n_obs

                pos_x = n_pos_x
                pos_y = n_pos_y

            elif act == "toggle":

                o_world_map = world_map.copy()
                n_obs, reward, terminated, truncated, _ = env.step(Actions.toggle)
                act_his.append(act)
                if terminated:
                    pos_x, pos_y, arrow = pos_m[env_id]
                    sys.exit()
                obj_intr_rec[env][0][get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)] += 1
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his)
                c_exp = sum_exp(args, n_exp, exp, act_his)

                exp = c_exp
                obs = n_obs

            elif act == "drop off":
                o_world_map = world_map.copy()
                n_obs, reward, terminated, truncated, _ = env.step(Actions.drop)
                act_his.append(act)
                if terminated:
                    pos_x, pos_y, arrow = pos_m[env_id]
                    sys.exit()
                else:
                    if not np.array_equal(n_obs, obs):
                        n_inv = 0
                obj_intr_rec[env][1][get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)] += 1
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, n_inv, act_his)
                c_exp = sum_exp(args, n_exp, exp, act_his)

                exp = c_exp
                obs = n_obs
                inv = n_inv
                
            elif act == "pick up":
                o_world_map = world_map.copy()
                n_obs, reward, terminated, truncated, _ = env.step(Actions.pickup)
                act_his.append(act)
                if terminated:
                    pos_x, pos_y, arrow = pos_m[env_id]
                    sys.exit()
                else:
                    if not np.array_equal(n_obs, obs):
                        n_inv = get_n_inv(args, n_obs, obs)
                obj_intr_rec[env][2][get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)] += 1
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, n_inv, act_his)
                c_exp = sum_exp(args, n_exp, exp, act_his)

                exp = c_exp
                obs = n_obs
                inv = n_inv

            if args.wandb:
                scn_table.add_data(wandb.Image(img), act_msg_s, act, n_exp, c_exp)
                env_table.add_data(str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", ""))
                obj_table.add_data(str(obj_intr_rec[env_id]), str(obj_view_rec[env_id]))
                world_map_table.add_data(str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", ""))
        
        env_view_ratio, env_step_ratio, env_memo_ratio, obj_intr_ratio, obj_view_ratio = get_ratios(args, env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec)


        # Environment close() due to all steps finished      
        env.close()

        # Log datas to the wandb
        if args.wandb:
            wandb.log({f"Table/Screenshot for Environment #{i}": scn_table})
            wandb.log({f"Table/Environment Record for Environment #{i}": env_table})
            wandb.log({f"Table/Object Record for Environment #{i}": obj_table})
            wandb.log({f"Table/World Map for Environment #{i}": world_map_table})

    wandb.finish()