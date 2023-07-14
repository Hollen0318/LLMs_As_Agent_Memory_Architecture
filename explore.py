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

# Function called by the OpenAI API to choose an action
def get_action(args, text):
    if args.log:
        print(f"\n################## Starting Deciding ##################\n")
        print(f"Prompt Message =\n\n{text}\n")
    write_log(f"\n################## Starting Deciding ##################\n")
    write_log(f"Prompt Message =\n\n{text}\n")
    act_obj_pair = {"0": "left", "1": "right", "2": "toggle",
                    "3": "forward", "4": "pick up", "5": "drop off"}
    if args.input:
        if args.log:
            print(f"{open(args.sys_msg).read()}")
            print(f"{text}")
            print(f"{open(args.fuc_msg).read()}\n")
        write_log(f"{open(args.sys_msg).read()}")
        write_log(f"{text}")
        write_log(f"{open(args.fuc_msg).read()}\n")
        valid = [str(i) for i in range(6)]
        while True:
            act = input("") 
            if act in valid:
                break
            else:
                continue
    else:
        valid = [i for i in range(6)]
        gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
        sys_msg = open(args.sys_msg).read()
        if args.goal:
            sys_msg += "\nYou will be prompted a goal specific to the environment.\n"
        msg = [{"role": "system", "content": sys_msg}]
        fuc_msg = open(args.fuc_msg).read()
        fuc = [{"name": "choose_act","description":fuc_msg,"parameters":{"type":"object", "properties":{"action":{"type":"integer", "description":"the action to take (in integer)","enum":[i for i in range(6)]}}}}]
        usr_msg = text     
        msg.append({"role": "user", "content": usr_msg})
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
    return act_obj_pair[act]

# Get the mapping list between 0,1,2,3 and environment names in a list
def get_env_id_mapping(args):
    file_name = args.env_id_maps
    id_mappings = []
    with open(file_name, "r") as file:
        for line in file:
            _, env_name = line.strip().split(", ")
            id_mappings.append(env_name)
    return id_mappings

# Getting the experience based on two text description and past actions 
def get_exp(args, reflect_hint, p_exp, act_his):
    if args.log:
        print(f"\n################## Starting Reflection ##################\n")
    write_log(f"\n################## Starting Reflection ##################\n")
    if args.input:
        if args.log:
            print(f"Prompt Message = \n\n{reflect_hint}")
        write_log(f"Prompt Message = \n\n{reflect_hint}")
        c_exp = input("Write your experience here")
    else:
        gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
        sys_msg = open(args.sys_msg).read()
        if args.goal:
            sys_msg += "You will be prompted a goal in the environment.\n"
        msg = [{"role": "system", "content": sys_msg}]
        usr_msg = reflect_hint
        if args.log:
            print(f"Prompt Message = \n\n{usr_msg}")
        write_log(f"Prompt Message = \n\n{usr_msg}")
        msg.append({"role": "user", "content": usr_msg})
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
            print(f"\n***************** Gained Experience *****************\n")
            print(f"{n_exp}")
            print(f"\n################## Starting Reviewing ##################\n")
        write_log(f"\n***************** Gained Experience *****************\n")
        write_log(f"{n_exp}")
        write_log(f"\n################## Starting Reviewing ##################\n")
        msg = [{"role": "system", "content": sys_msg}]
        usr_msg = write_sum_temp(args, p_exp, n_exp, act_his)
        if args.log:
            print(f"Prompt Message = \n\n{usr_msg}")
        write_log(f"Prompt Message = \n\n{usr_msg}")
        msg.append({"role": "user", "content": usr_msg})
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
        print(f"\n***************** Summarized Experience *****************\n")
        print(f"{c_exp}")
    write_log(f"\n***************** Summarized Experience *****************\n")
    write_log(f"{c_exp}")
    return n_exp, p_exp, c_exp

# Function to get the front object based on the observation
def get_fro_obj(args, obs):

    img = obs['image'].transpose(1,0,2)

    # The user location in the MiniGrid is always (view//2,view-1) where the x axis
    # points to the right and y axis points to the down direction 
    user_x, user_y = args.view - 1, args.view // 2

    for x in range(args.view):
        for y in range(args.view):
            # We record the object in front of the agent for future use
            if y == user_y and x == user_x - 1:
                fro_obj_l = img[x,y]

    obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
    col_idx = {0: "black", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
    sts_idx = {0: "open", 1: "closed", 2: "locked"}

    fro_obj_s = f"Your front object is {obj_idx[fro_obj_l[0]]} {col_idx[fro_obj_l[1]]} {sts_idx[fro_obj_l[2]]}"

    return fro_obj_s, fro_obj_l

# Get the saving path for the current argument setting
def get_path(args):
    # Test if the model is getting directions from input
    if args.input:
        dir_n = "INPUT"
    else:
        dir_n = "GPT"
    # Get today's date and format it as MM_DD_YYYY
    if args.all:
        env_names = "ALL"
    else:
        env_names = "_".join(args.envs)
    arg_list = ["seed", "gpt", "view", "goal", "static", "temp", "steps", "lim", "refresh"]
    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Combine them to create the full path
    full_path = os.path.join(dir_n, env_names, folder_name)
    return full_path

# Function to get re-spawn position (when seed = 23 only)
def get_pos_m(args):
    # read data from txt file
    with open(args.env_pos, 'r') as f:
        lines = f.readlines()

    # parse the data and create matrices
    pos_m = {}
    for line in lines:
        env_id, x, y, arrow = line.strip().split(',')
        pos_m[env_id] = (int(x), int(y), arrow)

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
def get_ratios(args, env_id, env_rec, obj_rec):
    env_view_r = np.sum(env_rec[env_id][0]) / env_rec[env_id][0].size * 100
    env_view_r_s = "{:.3f}%".format(env_view_r)
    env_intr_r = np.sum(env_rec[env_id][1]) / env_rec[env_id][1].size * 100
    env_intr_r_s = "{:.3f}%".format(env_intr_r)
    env_step_r = np.sum(env_rec[env_id][2]) / env_rec[env_id][2].size * 100
    env_step_r_s = "{:.3f}%".format(env_step_r)
    obj_view_r = np.sum(obj_rec[env_id][0]) / obj_rec[env_id][0].size * 100
    obj_view_r_s = "{:.3f}%".format(obj_view_r)
    obj_intr_r = np.sum(obj_rec[env_id][1]) / obj_rec[env_id][1].size * 100
    obj_intr_r_s = "{:.3f}%".format(obj_intr_r)
    
    return env_view_r, env_intr_r, env_step_r, obj_view_r, obj_intr_r, env_view_r_s, env_intr_r_s, env_step_r_s, obj_view_r_s, obj_intr_r_s

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
    with open(args.env_sizes, 'r') as f:
        lines = f.readlines()

    # parse the data and create matrices
    env_view_rec = {}
    obj_intr_rec = {}
    obj_view_rec = {}

    for line in lines:
        env_id, h, w = map(int, line.strip().split(','))
        env_view_rec[env_id] = np.zeros((h, w))
        obj_intr_rec[env_id] = np.array([[0 for i in range(11)] for j in range(3)])
        obj_view_rec[env_id] = np.array([0 for i in range(11)])

    return env_view_rec, obj_intr_rec, obj_view_rec

# Get the observation map for all environments, with 3-dimension (object, color, status) and height, width.
def get_world_maps(args):
    with open(args.env_sizes, 'r') as f:
        lines = f.readlines()

    # parse the data and create world maps
    world_map = {}
    for line in lines:
        env_id, h, w = map(int, line.strip().split(','))
        world_map[env_id] = np.empty((3, h, w), dtype = str)    
        # the three dimensions will be object, color and status, we intiialize them seperately now 
        world_map[env_id][0] = np.full((h, w), "0", dtype = str)
        world_map[env_id][1] = np.full((h, w), "-", dtype = str)
        world_map[env_id][2] = np.full((h, w), "-", dtype = str)

    return world_map

def link_rec_obs(args, env_id, env_rec, obj_rec, obs, pos):
    n_env_rec = env_rec.copy()
    n_obj_rec = obj_rec.copy()
    # 1. Updating the env record
    img = obs['image'].transpose(1,0,2)
    if args.log:
        print(f"\n################## Starting Updating ##################\n")
    write_log(f"\n################## Starting Updating ##################\n")
    # Update the env view record:
    # These are the indexes to record the position in img
    rel_x = 0
    rel_y = 0
    if obs['direction'] == 0:
        img_rel = np.rot90(img, k = 1, axes = (0, 1))
        for x in range(pos[0] - args.view // 2, pos[0] + args.view // 2 + 1):
            for y in range(pos[1], pos[1] + args.view):
                n_env_rec = update_view(args, img_rel, n_env_rec, env_id, x, y, rel_x, rel_y)
                rel_y += 1
            rel_y = 0
            rel_x += 1
        rel_x = 0
    elif obs['direction'] == 1:
        img_rel = np.rot90(img, k = 2, axes = (0, 1))
        for x in range(pos[0], pos[0] + args.view):
            for y in range(pos[1] - args.view // 2, pos[1] + args.view // 2 + 1):
                n_env_rec = update_view(args, img_rel, n_env_rec, env_id, x, y, rel_x, rel_y)
                rel_y += 1
            rel_y = 0
            rel_x += 1
        rel_x = 0
    elif obs['direction'] == 2:
        img_rel = np.rot90(img, k = 3, axes = (0, 1))
        for x in range(pos[0] - args.view // 2, pos[0] + args.view // 2 + 1):
            for y in range(pos[1] - args.view + 1, pos[1] + 1):
                n_env_rec = update_view(args, img_rel, n_env_rec, env_id, x, y, rel_x, rel_y)
                rel_y += 1
            rel_y = 0
            rel_x += 1
        rel_x = 0
    elif obs['direction'] == 3:
        img_rel = img.copy()
        for x in range(pos[0] - args.view + 1, pos[0] + 1):
            for y in range(pos[1] - args.view // 2, pos[1] + args.view // 2 + 1):
                n_env_rec = update_view(args, img_rel, n_env_rec, env_id, x, y, rel_x, rel_y)
                rel_y += 1
            rel_y = 0
            rel_x += 1
        rel_x = 0

    # Update the env step record:
    n_env_rec[env_id][2][pos] = 1

    # 2. Update the obj record

    # Update the obj view record
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            n_obj_rec[env_id][0][img[x,y][0]] = 1

    return n_env_rec, n_obj_rec

# Function to update the view based on the observation
def update_view(args, img_rel, env_rec, env_id, x, y, rel_x, rel_y):
    if x >= 0 and x < env_rec[env_id][0].shape[0] and y >= 0 and y < env_rec[env_id][0].shape[1]:
        if args.log:
            print(f"Updating view at x = {x} and y = {y}\nIn relative image this is x = {rel_x} and y = {rel_y}\n")
        write_log(f"Updating view at x = {x} and y = {y}\nIn relative image this is x = {rel_x} and y = {rel_y}\n")
        if img_rel[rel_x,rel_y][0] != 0:
            env_rec[env_id][0][x,y] = 1
    return env_rec

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

# Function to write into the act_temp.txt to hint an action
def write_act_temp(args, world_map, pos, obs, inv, env_id, act_his, c_exp, fro_obj_l):
    dir = obs['direction']
    dir_dic = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}
    dir_s = dir_dic[dir]

    img = obs['image'].transpose(1,0,2)
    act_his_s = ", ".join(act_his)
    with open(args.act_temp, 'r') as file:
        temp = file.read()

    obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
    col_idx = {0: "black", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
    sts_idx = {0: "open", 1: "closed", 2: "locked"}

    fro_obj_s = f"Your front object is {obj_idx[fro_obj_l[0]]} {col_idx[fro_obj_l[1]]} {sts_idx[fro_obj_l[2]]}"

    act_temp_s = temp.format(str(world_map), str(pos), dir_s, str(img), str(inv), str(env_id), act_his_s, c_exp, fro_obj_s)
    
    return act_temp_s

# Function to write the logging infos in to log save file
def write_log(save_path, text):
    # Open the file in append mode
    with open(os.path.join(save_path, f"log.txt"), "a") as file:
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
    
    with open(args.refl_temp, 'r') as file:
        temp = file.read()

    refl_temp_s = temp.format(str(o_world_map), str(o_pos), o_dir_s, str(o_img), str(o_inv), 
                             str(o_env_id), act, o_fro_obj_s, str(n_world_map), str(n_pos), 
                             n_dir_s, str(n_img), str(n_inv), str(n_env_id), n_fro_obj_s, 
                             n_act_his_s, str(args.lim))
    
    return refl_temp_s

# Function to write into the rpt_temp.txt to report the current progress.
def write_rpt_temp(args, env_view, env_intr, env_step, obj_view, obj_intr, pos, 
                   env_view_r_s, env_intr_r_s, env_step_r_s, obj_view_r_s, obj_intr_r_s):
    
    with open(args.rpt_temp, 'r') as file:
        temp = file.read()

    rpt_temp_s = temp.format(str(env_view), str(env_intr), str(env_step), 
                             str(obj_view), str(obj_intr), str(pos),
                             env_view_r_s, env_intr_r_s, env_step_r_s,
                             obj_view_r_s, obj_intr_r_s)
    
    return rpt_temp_s

# Function to write into the sum_temp.txt to hint an summarized experience based past & new experience
def write_sum_temp(args, o_exp, n_exp, act_his):

    with open(args.sum_temp, 'r') as file:
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
        default = "./utilities/act_temp.txt",
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
        default = ["1"]
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
        default = "./utilities/env_pos.txt"
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
        default = 0
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "LLM As Agent"
    )
    parser.add_argument(
        "--refl-temp",
        default = "./utilities/refl_temp.txt",
        type = str,
        help = "the location to load your reflect prompt message template"
    )
    parser.add_argument(
        "--rpt-temp",
        default = "./utilities/rpt_temp.txt",
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
        default = "./utilities/sum_temp.txt",
        type = str,
        help = "the location to load your summarization prompt message template"
    )
    parser.add_argument(
        "--sys-msg",
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
    parser.add_argument(
        "--refresh",
        type = int,
        default = 20,
        help = "the longest number of actionn historys"
    )
    parser.add_argument(
        "--forget-t",
        type = int,
        default = 10,
        help = "for how long will the agent starts to forget regions outside of view"
    )
    parser.add_argument(
        "--forget-r",
        type = float,
        default = 0.8,
        help = "for how likely the agent will forget regions esceeding the forget-t"
    )

    args = parser.parse_args()
    save_path = get_path(args)

    # Start running the specified environment(s), for each one, it has limited steps, whether it uses an existing
    # experience or an evolving experience will be dependent on the arguments. 
    if args.log:
        print(f"################## Starting Experiment ##################\n")
        print(f"Configurations are:\n{args}")
    write_log(save_path, f"################## Starting Experiment ##################\n")
    write_log(save_path, f"Configurations are:\n{args}")

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
    
    # index to record the action & label the image
    act_idx = 0

    # get the environment list based on --all or --envs, if --all then replace args.envs as all environments
    if args.all:
        args.envs = [str(i) for i in range(61)]
    if args.log:
        print(f"\n################## System Message ##################\n")
        print(f"{open(args.sys_msg).read()}")    
    write_log(f"{open(args.sys_msg).read()}")
    write_log(f"\n################## System Message ##################\n")

    # Get the observation map for all environments, with 3-dimension (object, color, status) and height, width.
    world_map = get_world_maps(args)
    # Get the two record matrix for all environments, with environment and object level
    env_view_rec, obj_intr_rec, obj_view_rec = get_rec(args)
    # The environment ID and enviornment name mapping list
    envs_id_mapping = get_env_id_mapping(args)
    # Get the position mapping for all environments, which include the x, y (in integer) and the direction → string
    pos_m = get_pos_m(args)
    # Iterate over all the environment
    for i in args.envs:
        # For every new environment, the inventory is always 0 (empty)
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
            rec_table = wandb.Table(columns = ["env_view", "env_step", "pos", "dir", "act", "obj_view", "obj_intr"])
        
        # Initilize the environment
        obs, state = env.reset(seed=args.seed)

        # env_rec, obj_rec = link_rec_obs(args, int(n_env_id), env_rec, obj_rec, obs, pos)
        # if args.wandb:
        #     dir_dic = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}
        #     rec_table.add_data(str(env_rec[int(n_env_id)][0]), str(env_rec[int(n_env_id)][1]), 
        #                        str(env_rec[int(n_env_id)][2]), str(pos), dir_dic[obs['direction']],
        #                        "start", str(obj_rec[int(n_env_id)][0]), str(obj_rec[int(n_env_id)][1]))
        
        # Iterate the agent exploration within the limit of args.steps
        for j in range(args.steps):

            # Link the obs to the record table, i.e. we record what agent's has seen in environment view and object view level
            update_rec(args, obs, env_view_rec, obj_view_rec)
            # With the environment record, and observation, we update the world map, we include env_view_rec 
            # because it will be used to determine which areas are forgotten 
            update_world_map(args, obs, world_map, env_view_rec)    
            # With new world map, we can then generate the texts based on the world map.
            act_hint = act_hint_gen(args, world_map)
            # We get the action from the act_hint, the act is a string format like "pick up"
            act = get_action(act_hint)
            # With the new act
            # get five ratios measuring the exploration ratio
            env_view_r, env_intr_r, env_step_r, obj_view_r, obj_intr_r, env_view_r_s, env_intr_r_s, env_step_r_s, obj_view_r_s, obj_intr_r_s = get_ratios(args, int(n_env_id), env_rec, obj_rec)

            # We refresh the action history every args.refresh run to avoid too large action space
            if len(act_his) >= args.refresh:
                act_his = act_his[1:]

            # gain the fro_obj
            fro_obj_s, fro_obj_l = get_fro_obj(args, obs)

            # write into the act template to obtain a action hint message
            act_hint = write_act_temp(args, env_rec[int(n_env_id)][0], pos, obs, inv, int(n_env_id), act_his, exp, fro_obj_l)

            # get an action in text e.g. forward, pick up
            act = get_action(args, act_hint)

            if args.log:
                print(f"***************** Gained Action *****************\n")
                print(f"You have choose to do \"{act}\"")
            write_log(f"***************** Gained Action *****************\n")
            write_log(f"You have choose to do \"{act}\"")

            # using the action to determine the new inventory and MiniGrid action object
            n_inv, act_obj = cvt_act(args, inv, act, fro_obj_l)

            if args.disp:
                scn = pyautogui.screenshot()
                scn.save(os.path.join(save_path, f"env_{i}_action_{act_idx}_{act}.png"))
            else:
                # make a screenshot on the gym
                img_array = env.render()
                img = Image.fromarray(img_array)
                img.save(os.path.join(save_path, f"env_{i}_action_{act_idx}_{act}.png"))

            # take the action returned either by api or user
            n_obs, reward, terminated, truncated, _ = env.step(act_obj)

            if terminated:

                # We restart current environment if terminated like stepping into the lava or finish the goal
                act_his = []

                # TODO Add the termination process
                # Get the respawn position

                pos = pos_m[int(i)]

            else:
                # get the new act_his list
                n_act_his = act_his.copy()
                n_act_his.append(act)
                # get the new fro_obj
                n_fro_obj_s, n_fro_obj_l = get_fro_obj(args, n_obs)

                # We want the old observation and new observation to generate experience
                n_pos, n_env_rec, n_obj_rec = update_rec(args, env_rec, obj_rec, int(n_env_id), act, pos, n_obs, fro_obj_l)
                n_env_rec, n_obj_rec = link_rec_obs(args, int(n_env_id), n_env_rec, n_obj_rec, n_obs, n_pos)

                # write into the reflect template to obtain a reflection hint message
                reflect_hint = write_refl_temp(args, env_rec[o_env_id][0], pos, obs, inv, 
                                               o_env_id, act, fro_obj_s, 
                                               n_env_rec[int(n_env_id)][0], n_pos, n_obs, n_inv, 
                                               n_env_id, n_fro_obj_s, n_act_his)

                # get a new experience
                if args.static:
                    continue

                else:
                    n_exp, p_exp, c_exp = get_exp(args, reflect_hint, exp, n_act_his)
                    with open(os.path.join(save_path, f"env_{i}_action_{act_idx}_{act}.txt"), "w") as f:
                        f.write(f"New experience:\n\n{n_exp}\n")
                        f.write(f"\nPast experience:\n\n{p_exp}\n")
                        f.write(f"Summarized experiene:\n\n{c_exp}\n")

                if args.wandb:

                # log everything to the wandb    
                    scn_table.add_data(wandb.Image(img), str(obs), act_hint, str(pos), act, n_exp, c_exp)
                    dir_dic = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}
                    rec_table.add_data(str(env_rec[int(n_env_id)][0]), str(env_rec[int(n_env_id)][1]), 
                                       str(env_rec[int(n_env_id)][2]), str(n_pos), dir_dic[obs['direction']],
                                       act, str(obj_rec[int(n_env_id)][0]), str(obj_rec[int(n_env_id)][1]))
                    metrics = {
                        "env_view_ratio": env_view_r,
                        "env_intr_ratio": env_intr_r,
                        "env_step_ratio": env_step_r,
                        "obj_view_ratio": obj_view_r,
                        "obj_intr_ratio": obj_intr_r
                    }

                    # Log the metrics
                    wandb.log(metrics)

                # Print the report by writing into the file rpt_temp.txt
                report_text = write_rpt_temp(args, env_rec[int(n_env_id)][0], env_rec[int(n_env_id)][1], env_rec[int(n_env_id)][2], 
                                             obj_rec[int(n_env_id)][0], obj_rec[int(n_env_id)][1], pos, 
                                             env_view_r_s, env_intr_r_s, env_step_r_s, obj_view_r_s, obj_intr_r_s)
                
                # Print the records and write them to the log files
                if args.log:
                    print(f"\n***************** Records *****************\n")
                    print(f"{report_text}\n")
                write_log(f"\n***************** Records *****************\n")
                write_log(f"{report_text}\n")
                
                # Update the observation into the new observation to be used by later deciding
                obs = n_obs
                
                # Update the record to be the new rec
                # update the env & obj record matrix 
                pos, env_rec, obj_rec = n_pos, n_env_rec, n_obj_rec

                # Increment the action index
                act_idx += 1

                # Update the experience into the combined experience to be used by later deciding
                exp = c_exp

                # Update the inv to be the n_inv
                inv = n_inv

                # Update the act history to the n_act_his
                act_his = n_act_his.copy()

        # Update the environment ID into the new one
        o_env_id = n_env_id

        # Environment close() due to all steps finished      
        env.close()

        # Log datas to the wandb
        if args.wandb:
            wandb.log({f"Table/Screenshot for Environment #{i}": scn_table})
            wandb.log({f"Table/Record for Environment #{i}": rec_table})
            
    wandb.finish()