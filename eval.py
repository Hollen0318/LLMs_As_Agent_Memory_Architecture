# Packages
import pandas as pd
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
import numpy as np
import re

# Get the observation representation description, to aid in the decision making
def get_desc(args, env_id, world_map, inv, exp, pos_x, pos_y, arrow, lim):
    desc = f"This is environment #{str(env_id)}\n"
    global save_path
    global gpt_map
    global utilities
    write_log(args, save_path, f"\n\n################## Start Describing ##################\n\n")
    
    # To get an action, we need first to fill the sys_msg.txt with the args.refresh and use it as system message
    global sys_msg_s    
    # Then we need the observation message, which we will fill the act_temp.txt
    obj_map_s = np.array2string(world_map[env_id][0]).replace("'", "").replace("\"","")
    col_map_s = np.array2string(world_map[env_id][1]).replace("'", "").replace("\"","")
    sta_map_s = np.array2string(world_map[env_id][2]).replace("'", "").replace("\"","")
    obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
    if inv == 0:
        inv_s = f"You are not holding anything"
    else:
        inv_s = f"You are holding a {obj_idx[inv]}"
    if lim == 0:
        desc_msg = utilities['eval_desc_msg_no_e']
        desc_msg_s = desc_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, str(lim))
    else:
        desc_msg = utilities['eval_desc_msg_e']
        desc_msg_s = desc_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, exp, str(lim))
    arrow_s = arrow[0].lower() + arrow[1:]
    msg = [{"role": "system", "content": "Your mission is to understand deeply and follow a interpretation format to describe a text-based environment as follows:"}]
    msg.append({"role": "user", "content": sys_msg_s})
    if lim == 0:
        msg.append({"role": "assistant", "content": "Sure, give me the real world map, inventory, I will describe about it to aid in fulfilling the mission"})
    else:
        msg.append({"role": "assistant", "content": "Sure, give me the real world map, inventory and experience, I will describe about it to aid in fulfilling the mission"})
    msg.append({"role": "user", "content": desc_msg_s})
    write_log(args, save_path, f"Prompt message = \n\n{desc_msg_s}")
    retry_delay = args.rty_dly  # wait for 1 second before retrying initially
    while True:
        try:
            rsp = openai.ChatCompletion.create(
                model = gpt_map[args.gpt],
                messages = msg,
                temperature = args.temp,
                max_tokens = lim
            )
            desc += rsp["choices"][0]["message"]["content"]
            break
        except Exception as e:
            write_log(args, save_path, f"Caught an error: {e}\n")
            time.sleep(retry_delay)
            retry_delay *= 2  # double the delay each time we retry
    write_log(args, save_path, f"\n\n***************** Gained Description *****************\n\n")
    write_log(args, save_path, f"{desc}")
    return desc

def get_env_id_mapping(args):
    global utilities
    id_mappings = []
    for env in utilities['env_id_maps'].strip().split("\n"):
        _, env_name = env.strip().split(", ")
        id_mappings.append(env_name)
    return id_mappings

def get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow):
    if arrow == "Up":
        return int(world_map[env_id][0][pos_x - 1][pos_y])
    elif arrow == "Down":
        return int(world_map[env_id][0][pos_x + 1][pos_y])
    elif arrow == "Left":
        return int(world_map[env_id][0][pos_x][pos_y - 1])
    elif arrow == "Right":
        return int(world_map[env_id][0][pos_x][pos_y + 1])
    
def get_front_col(args, env_id, world_map, pos_x, pos_y, arrow):
    if arrow == "Up":
        return int(world_map[env_id][1][pos_x - 1][pos_y])
    elif arrow == "Down":
        return int(world_map[env_id][1][pos_x + 1][pos_y])
    elif arrow == "Left":
        return int(world_map[env_id][1][pos_x][pos_y - 1])
    elif arrow == "Right":
        return int(world_map[env_id][1][pos_x][pos_y + 1])
    
def get_front_sta(args, env_id, world_map, pos_x, pos_y, arrow):
    if arrow == "Up":
        return int(world_map[env_id][2][pos_x - 1][pos_y])
    elif arrow == "Down":
        return int(world_map[env_id][2][pos_x + 1][pos_y])
    elif arrow == "Left":
        return int(world_map[env_id][2][pos_x][pos_y - 1])
    elif arrow == "Right":
        return int(world_map[env_id][2][pos_x][pos_y + 1])

def get_goals_by_env_and_level(args, data, env_id, level_name):
    global save_path
    for environment in data["environments"]:
        if environment["id"] == env_id:
            env_name = environment["name"]
            for level in environment["level"]:
                if level["level"] == level_name:
                    return env_name, level["goals"]
    write_log(args, save_path, "goals not found")
    return None, None

# Get the new inventory based on difference between o_obs and n_obs
def get_n_inv(args, n_obs, o_obs):
    n_obs_img_obj = n_obs['image'].transpose(1,0,2)[:, :, 0]
    o_obs_img_obj = o_obs['image'].transpose(1,0,2)[:, :, 0]
    indices = np.where(n_obs_img_obj != o_obs_img_obj)
    return o_obs_img_obj[indices[0][0]][indices[1][0]]

# Get the saving path for the current argument setting
def get_path(args, env_id, lim):
    dir_n = "EVAL"
    timestamp = datetime.now().strftime(r"%Y-%m-%d %H-%M-%S")
    arg_list = ["seed", "gpt", "memo", "temp", "view"]
    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Combine them to create the full path
    full_path = os.path.join(dir_n, env_id, str(lim), folder_name, str(timestamp))
    
    return full_path

# Function to get re-spawn position (when seed = 23 only)
def get_pos_m(seed):
    global utilities
    key_name = "env_pos_" + str(seed)
    # parse the data and create matrices
    pos_m = {}
    for env in utilities[key_name].strip().split("\n"):
        env_id, x, y, arrow = env.strip().split(', ')
        pos_m[int(env_id)] = (int(x), int(y), arrow)
    return pos_m

def get_rec(seed):
    global utilities
    # read data from txt file
    key_name = f'env_sizes_{str(seed)}'
    # parse the data and create matrices
    env_view_rec = {}
    env_step_rec = {}
    env_memo_rec = {}
    obj_intr_rec = {}
    obj_view_rec = {}

    for env in utilities[key_name].strip().split("\n"):
        env_id, h, w = map(int, env.strip().split(','))
        env_view_rec[env_id] = np.zeros((h, w))
        env_step_rec[env_id] = np.zeros((h, w))
        env_memo_rec[env_id] = np.zeros((h, w))
        obj_intr_rec[env_id] = np.array([[0 for i in range(11)] for j in range(3)])
        obj_view_rec[env_id] = np.array([0 for i in range(11)])

    return env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec

def get_reason(args, world_map, inv, obs, exp, desc, pos_x, pos_y, arrow, env_id, lim):
    global save_path
    global gpt_map
    global sys_msg_s
    global utilities
    write_log(args, save_path, f"\n\n################## Start Deciding ##################\n\n")
    act_obj_pair = {"0": "left", "1": "right", "2": "toggle",
                    "3": "forward", "4": "pick up", "5": "drop off"}
    # Then we need the observation message
    if lim == 0:
        reason_msg = utilities['eval_reason_msg_no_e']
    else:
        reason_msg = utilities['eval_reason_msg_e']
    obj_map_s = np.array2string(world_map[env_id][0]).replace("'", "").replace("\"","")
    col_map_s = np.array2string(world_map[env_id][1]).replace("'", "").replace("\"","")
    sta_map_s = np.array2string(world_map[env_id][2]).replace("'", "").replace("\"","")
    obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
    if inv == 0:
        inv_s = f"You are not holding anything"
    else:
        inv_s = f"You are holding a {obj_idx[inv]}"
    arrow_s = arrow[0].lower() + arrow[1:]
    if lim == 0:
        reason_msg_s = reason_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, desc, str(lim))
    else:
        reason_msg_s = reason_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, exp, desc, str(lim))

    msg = [{"role": "system", "content":  "You mission to be an agent that's about to explore a text based world, with environment observation representation provided by user."}]
    msg.append({"role": "user", "content": sys_msg_s})
    if lim == 0:
        msg.append({"role": "assistant", "content": "Sure, give me the real observation in world map, inventory and I will decide to make one move or multiple step moves."})
    else:
        msg.append({"role": "assistant", "content": "Sure, give me the real observation in world map, inventory, experience and I will decide to make one move or multiple step moves."})
    msg.append({"role": "user", "content": reason_msg_s})
    write_log(args, save_path, f"Prompt message = \n\n{reason_msg_s}")
    retry_delay = args.rty_dly  # wait for 1 second before retrying initially
    while True:
        try:
            rsp = openai.ChatCompletion.create(
                model=gpt_map[args.gpt],
                messages=msg,
                temperature = args.temp,
                max_tokens = args.reason
            )
            reason = rsp["choices"][0]["message"]["content"]
            break
        except Exception as e:
            write_log(args, save_path, f"Caught an error: {e}\n")
            time.sleep(retry_delay)
            retry_delay *= 2  # double the delay each time we retry
    write_log(args, save_path, f"\n\n***************** Gained Reason *****************\n\n")
    write_log(args, save_path, f"{reason}")
    return reason, reason_msg_s

# Get the observation map for all environments, with 3-dimension (object, color, status) and height, width.
def get_world_maps(seed):
    # parse the data and create world maps
    key_name = f'env_sizes_{str(seed)}'
    world_map = {}
    global utilities
    for env in utilities[key_name].strip().split("\n"):
        env_id, h, w = map(int, env.strip().split(", "))
        world_map[env_id] = np.empty((3, h, w), dtype = object)    
        # the three dimensions will be object, color and status, we intiialize them seperately now 
        world_map[env_id][0] = np.full((h, w), "-", dtype = object)
        world_map[env_id][1] = np.full((h, w), "-", dtype = object)
        world_map[env_id][2] = np.full((h, w), "-", dtype = object)

    return world_map

def left_arrow(arrow):
    r_arrow = ""
    if arrow == "Up":
        r_arrow = "Left"
    elif arrow == "Down":
        r_arrow = "Right"
    elif arrow == "Left":
        r_arrow = "Down"
    elif arrow == "Right":
        r_arrow = "Up"
    return r_arrow

def load_dict_from_json(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)

def locate_exp(args, x, y, gpt, memo, view, temp):
    global save_path
    base_dir = f"GPT/{x}/"
    base_pattern = f"desc_{y}_goal_False_gpt_{gpt}_lim_{y}_memo_{memo}_reason_{y}_seed_23_steps_"
    step_pattern = re.compile(f"steps_(\d+)_temp_{temp}_view_{view}")
    
    max_k = -1
    max_k_file_path = ""
    
    # Discover directories matching the base pattern
    steps_dirs = [d for d in os.listdir(base_dir) if d.startswith(base_pattern) and step_pattern.search(d)]
    for steps_dir in steps_dirs:
        # Get all date-named directories inside the steps_dir
        date_dirs = [d for d in os.listdir(os.path.join(base_dir, steps_dir)) if re.match(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", d)]
        for date_dir in date_dirs:
            # For each date directory, find the highest k value
            for file in os.listdir(os.path.join(base_dir, steps_dir, date_dir)):
                match = re.match(r"env_" + str(x) + r"_idx_(\d+)_exp_", file)
                if match:
                    k_val = int(match.group(1))
                    if k_val > max_k:
                        max_k = k_val
                        max_k_file_path = os.path.join(base_dir, steps_dir, date_dir, file)
    
    # Return content of the identified file
    if max_k_file_path:
        with open(max_k_file_path, 'r') as file:
            write_log(args, save_path, f"Loaded experience = {file.read()}\n\nThe path is {max_k_file_path}\n\nThe max_k experience length is {max_k}")
    else:
        write_log(args, save_path, "File not found.")

def right_arrow(arrow):
    r_arrow = ""
    if arrow == "Up":
        r_arrow = "Right"
    elif arrow == "Down":
        r_arrow = "Left"
    elif arrow == "Left":
        r_arrow = "Up"
    elif arrow == "Right":
        r_arrow = "Down"
    return r_arrow

# Update the position based on the arrow and previous position    
def update_pos(pos_x, pos_y, arrow):
    n_pos_x, n_pos_y = pos_x, pos_y
    if arrow == "Up":
        n_pos_x -= 1
    elif arrow == "Down":
        n_pos_x += 1
    elif arrow == "Left":
        n_pos_y -= 1
    elif arrow == "Right":
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
    global save_path
    # With the new obs, we should first update the env_memo_rec, as it will determine which parts of world map will show
    p_obj, p_col, p_sta = world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y]
    if arrow == "Right":
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

        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    elif arrow == "Up":
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
        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    elif arrow == "Down":
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
        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    elif arrow == "Left":
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
        world_map[env_id][0][pos_x][pos_y] = arrow
        world_map[env_id][1][pos_x][pos_y] = arrow
        world_map[env_id][2][pos_x][pos_y] = arrow
        env_step_rec[env_id][pos_x][pos_y] += 1
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
        # env_memo_rec[env_id][pos_x][pos_y] = args.memo
    # We update all the positions to -, and return the last position so can be used in the experience comparison
    for row in range(env_memo_rec[env_id].shape[0]):
        for col in range(env_memo_rec[env_id].shape[1]):
            if env_memo_rec[env_id][row][col] == 0:
                world_map[env_id][0][row][col] = "-"
                world_map[env_id][1][row][col] = "-"
                world_map[env_id][2][row][col] = "-"
    write_log(args, save_path, f"\n################## Debugging Position ##################\n")
    write_log(args, save_path, f"\nthe arrow is {arrow}, pos_x, pos_y, {str(pos_x)} {str(pos_y)}\n")
    return p_obj, p_col, p_sta

# Function to write the logging infos in to log save file
def write_log(args, save_path, text):
    if args.log:
        print(text)
    # Open the file in append mode
    with open(os.path.join(save_path, f"log.txt"), "a", encoding='utf-8') as file:
        # Write the strings to the file
        file.write(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--API-key",
        default = "./utilities/API/API_KEY",
        type = str,
        help = "the location to load your OpenAI API Key"
    )
    parser.add_argument(
        "--end",
        type = int,
        default = 100,
        help = "the end desc, lim, reason tokens"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "list of environment names like 0, 1, 2",
        default = ["0"]
    )
    parser.add_argument(
        "--eval",
        type = str,
        default = "./utlitilies/eval_envs.json",
        help = "the path to load your evaluation envs json"
    )
    parser.add_argument(
        "--goals",
        type = str,
        default = "./utilities/goals.json",
        help = "the path to load the goals json file"
    )
    parser.add_argument(
        "--gap",
        type = int, 
        help = "the incremental amount between the experience limit",
        default = 50
    )
    parser.add_argument(
        "--gpt",
        type = str,
        choices = ["3", "4"],
        help = "the version of gpt, type version number like 3 or 4",
        default = "3"
    )
    parser.add_argument(
        "--level",
        type = str,
        chocies = ["very easy", "easy", "medium", "hard", "very hard"]
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
        default = 10
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "LLM As Agent"
    )
    parser.add_argument(
        "--respawn",
        type = int,
        default = 5,
        help = "how many times we let agent respawn at different locations to evaluate"
    )
    parser.add_argument(
        "--rty-dly",
        type = int,
        default = 5,
        help = "the number of seconds to delay when in OpenAI API Calling"
    )
    parser.add_argument(
        "--seed",
        type = int,
        help = "random seed for reproducing results",
        default = 23
    )
    parser.add_argument(
        "--start",
        type = int,
        default = 50,
        help = "the start desc, lim, reason tokens"
    )
    parser.add_argument(
        "--steps",
        type = int,
        default = 100,
        help = "the total number of steps agent allowed to complete the goal"
    )
    parser.add_argument(
        "--temp",
        type = float,
        default = 0.7,
        help = "the temprature used by the OpenAI API"
    )
    parser.add_argument(
        "--utilities",
        type = str,
        default = "utilities/utilities.json",
        help = "the path to load your utilities JSON file storing all texts, environment name, start position etc"
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

    if args.wandb:
        wandb.init(
            project = args.prj_name,
            name = datetime.now().strftime(r"Eval %Y-%m-%d %H:%M:%S"),
            config = vars(args)
        )

    gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
    # Get the two record matrix for all environments, with environment and object level

    utilities = load_dict_from_json(args.utilities)

    # The environment ID and enviornment name mapping list
    envs_id_mapping = get_env_id_mapping(args)

    # Dictionary of environment being evaluated
    eval_envs = load_dict_from_json(args.utilities)

    goals_dict = load_dict_from_json(args.goals)

    for env_id in args.envs:
        # Complete non experience evaluation first
        env_id = int(env_id)
        env_name = envs_id_mapping[env_id]
        env: MiniGridEnv = gym.make(
            id = env_name,
            render_mode = "rgb_array",
            agent_view_size = args.view,
            screen_size = args.screen
        )
        write_log(args, save_path, f"Loading environment = {env_name}")
        if env_id == 0:
            for seed in range(20, 25):
                # Get the position mapping for all environments, which include the x, y (in integer) and the direction Right string
                pos_m = get_pos_m(seed)
                world_map = get_world_maps(seed)
                env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec = get_rec(seed)
                sys_msg= utilities['eval_sys_msg_no_e']
                sys_msg_s = sys_msg.format(str(world_map[env_id][0].shape[0]), str(world_map[env_id][0].shape[1]))
                inv = 0
                pos_x, pos_y, arrow = pos_m[env_id]
                # Initilize the environment
                obs, state = env.reset(seed=seed)
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                eval_desc_msg_no_e_s = get_desc(args, env_id, world_map, inv, exp, pos_x, pos_y, arrow, 0)
                reason, reason_msg_s = get_reason(args, world_map, inv, obs, exp, eval_desc_msg_no_e_s, pos_x, pos_y, arrow, env_id, 0)
                
        for lim in range(args.start, args.end + 1, args.gap):
            save_path = get_path(args, env_id, lim)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            exp = locate_exp(args, env_id, lim, args.gpt, args.memo, args.view, args.temp)
            if env_id == 0:
                for res in range(args.respawn):
                