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

# Function to return what GPT returns in sring format
def choose_act(action):
    return action

def count_tokens(string):
    # Define a pattern that matches words composed of alphanumeric characters
    pattern = r'\w+'
    
    # Find all tokens that match the pattern
    tokens = re.findall(pattern, string)
    
    # Return the number of tokens
    return len(tokens)

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

def execute_action(args, act):
    global world_map
    global arrow
    global env
    global exp
    global obs
    global pos_x
    global pos_y
    global env_step_rec
    global env_memo_rec
    global env_view_rec
    global obj_view_rec
    global obj_intr_rec
    global env_id
    global act_his
    global inv
    global p_obj
    global p_col
    global p_sta
    global c_exp
    global n_exp
    global n_arrow
    global n_obs
    global n_pos_x
    global n_pos_y
    global act_his
    global terminated
    global world_map

    if act == "left":

        # We update the world map, env view, env memo, obj_view, act history, arrow
        n_arrow = left_arrow(arrow)
        o_world_map = {}
        o_world_map[env_id] = world_map[env_id].copy()
        o_world_map[env_id][0] = world_map[env_id][0].copy()
        o_world_map[env_id][1] = world_map[env_id][1].copy()
        o_world_map[env_id][2] = world_map[env_id][2].copy()
        n_obs, reward, terminated, truncated, _ = env.step(Actions.left)
        act_his.append(act)
        _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, n_arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, n_arrow)
        arrow = n_arrow
        c_exp = sum_exp(args, n_exp, exp, act_his)

        exp = c_exp
        obs = n_obs

    elif act == "right":

        n_arrow = right_arrow(arrow)
        o_world_map = {}
        o_world_map[env_id] = world_map[env_id].copy()
        o_world_map[env_id][0] = world_map[env_id][0].copy()
        o_world_map[env_id][1] = world_map[env_id][1].copy()
        o_world_map[env_id][2] = world_map[env_id][2].copy()
        n_obs, reward, terminated, truncated, _ = env.step(Actions.right)
        act_his.append(act)
        _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, n_arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, n_arrow)
        arrow = n_arrow
        c_exp = sum_exp(args, n_exp, exp, act_his)

        exp = c_exp
        obs = n_obs

    elif act == "forward":
        o_world_map = {}
        o_world_map[env_id] = world_map[env_id].copy()
        o_world_map[env_id][0] = world_map[env_id][0].copy()
        o_world_map[env_id][1] = world_map[env_id][1].copy()
        o_world_map[env_id][2] = world_map[env_id][2].copy()
        n_obs, reward, terminated, truncated, _ = env.step(Actions.forward)
        act_his.append(act)
        if terminated:
            # For each new environment, the inventory is always 0
            inv = 0
            # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
            n_pos_x, n_pos_y, n_arrow = pos_m[env_id]
            # Initilize the environment
            obs, state = env.reset(seed=args.seed)
            front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
            world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
            # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
            p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            if front_obj == 8:
                n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
            elif front_obj == 9:
                n_exp = "You touched a lava and dead, respawn at the start place, try again!"
            elif front_obj == 6:
                n_exp = "You are killed stepping towards this ball"
            c_exp = sum_exp(args, n_exp, exp, act_his)
            exp = c_exp
            pos_x = n_pos_x
            pos_y = n_pos_y
            arrow = n_arrow
            return 
        else:
            front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
            front_col = get_front_col(args, env_id, world_map, pos_x, pos_y, arrow)
            front_sta = get_front_sta(args, env_id, world_map, pos_x, pos_y, arrow)

            if front_obj == 1 or front_obj == 3:
                n_pos_x, n_pos_y = update_pos(pos_x, pos_y, arrow)
            elif front_obj == 4 and front_sta == 0:
                n_pos_x, n_pos_y = update_pos(pos_x, pos_y, arrow)
            else:
                n_pos_x, n_pos_y = pos_x, pos_y
        write_log(args, save_path, f"\n*************************************************\n\nDoing forward, the p_obj and p_col, p_sta is {p_obj} {p_col} {p_sta } *************************************\n")
        world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
        p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his, pos_x, pos_y, n_pos_x, n_pos_y, arrow, arrow)
        c_exp = sum_exp(args, n_exp, exp, act_his)

        exp = c_exp
        obs = n_obs

        pos_x = n_pos_x
        pos_y = n_pos_y

    elif act == "toggle":
        o_world_map = {}
        o_world_map[env_id] = world_map[env_id].copy()
        o_world_map[env_id][0] = world_map[env_id][0].copy()
        o_world_map[env_id][1] = world_map[env_id][1].copy()
        o_world_map[env_id][2] = world_map[env_id][2].copy()
        front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
        if front_obj != 7:
            n_obs, reward, terminated, truncated, _ = env.step(Actions.toggle)
        else:
            n_obs, reward, terminated, truncated, _ = obs, 0.0, False, False, _
        act_his.append(act)
        if terminated:
            # For each new environment, the inventory is always 0
            inv = 0
            # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
            n_pos_x, n_pos_y, n_arrow = pos_m[env_id]
            # Initilize the environment
            obs, state = env.reset(seed=args.seed)
            world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
            # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
            _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
            c_exp = sum_exp(args, n_exp, exp, act_his)
            exp = c_exp
            pos_x = n_pos_x
            pos_y = n_pos_y
            arrow = n_arrow
            return
        write_log(args, save_path, f"The front object being interacted with is {front_obj}")
        obj_intr_rec[env_id][0][front_obj] += 1
        _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, arrow)
        c_exp = sum_exp(args, n_exp, exp, act_his)

        exp = c_exp
        obs = n_obs

    elif act == "drop off":
        o_world_map = {}
        o_world_map[env_id] = world_map[env_id].copy()
        o_world_map[env_id][0] = world_map[env_id][0].copy()
        o_world_map[env_id][1] = world_map[env_id][1].copy()
        o_world_map[env_id][2] = world_map[env_id][2].copy()
        n_obs, reward, terminated, truncated, _ = env.step(Actions.drop)
        act_his.append(act)
        if terminated:
            # For each new environment, the inventory is always 0
            inv = 0
            # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
            n_pos_x, n_pos_y, n_arrow = pos_m[env_id]
            # Initilize the environment
            obs, state = env.reset(seed=args.seed)
            front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
            world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
            # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
            _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
            c_exp = sum_exp(args, n_exp, exp, act_his)
            exp = c_exp
            pos_x = n_pos_x
            pos_y = n_pos_y
            arrow = n_arrow
            return 
        else:
            if not np.array_equal(n_obs['image'].transpose(1,0,2), obs['image'].transpose(1,0,2)):
                n_inv = 0
        front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
        write_log(args, save_path, f"The front object being interacted with is {front_obj}")
        obj_intr_rec[env_id][1][front_obj] += 1
        _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, n_inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, arrow)
        c_exp = sum_exp(args, n_exp, exp, act_his)

        exp = c_exp
        obs = n_obs
        inv = n_inv
        
    elif act == "pick up":
        o_world_map = {}
        o_world_map[env_id] = world_map[env_id].copy()
        o_world_map[env_id][0] = world_map[env_id][0].copy()
        o_world_map[env_id][1] = world_map[env_id][1].copy()
        o_world_map[env_id][2] = world_map[env_id][2].copy()
        n_obs, reward, terminated, truncated, _ = env.step(Actions.pickup)
        act_his.append(act)
        if terminated:
            # For each new environment, the inventory is always 0
            inv = 0
            # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
            n_pos_x, n_pos_y, n_arrow = pos_m[env_id]
            # Initilize the environment
            obs, state = env.reset(seed=args.seed)
            front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
            world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
            # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
            _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
            c_exp = sum_exp(args, n_exp, exp, act_his)
            exp = c_exp
            pos_x = n_pos_x
            pos_y = n_pos_y
            arrow = n_arrow
            return 
        else:
            if not np.array_equal(n_obs['image'].transpose(1,0,2), obs['image'].transpose(1,0,2)):
                n_inv = get_n_inv(args, n_obs, obs)
            else:
                n_inv = inv
        front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
        write_log(args, save_path, f"The front object being interacted with is {front_obj}")
        obj_intr_rec[env_id][2][front_obj] += 1
        _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, n_inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, arrow)
        c_exp = sum_exp(args, n_exp, exp, act_his)

        exp = c_exp
        obs = n_obs
        inv = n_inv

    # We refresh the action history every args.refresh run to avoid too large action space
    if len(act_his) >= args.memo:
        act_his = act_his[1:]

def get_action(args, reason):
    global save_path
    global gpt_map
    global utilities
    write_log(args, save_path, f"\n\n################## Start Deciding ##################\n\n")
    act_obj_pair = {"0": "left", "1": "right", "2": "toggle",
                    "3": "forward", "4": "pick up", "5": "drop off"}
    if args.input:
        # A demo action when using input
        act = input(f"""Choose an action or a mutiple step actions plan from the following: 
(0) left: turn your view to the left object
(1) right: turn your view to the right object
(2) toggle: toggle the object in front of you
(3) forward: move yourself to the front object if it's empty or opened
(4) pick up: pick up the front object if it can be picked up
(5) drop off: drop off the current inventory if front is empty\n\n""")
        return act_obj_pair[act]
    else:
        act_msg = utilities['act_msg']
        act_msg_s = act_msg.format(reason)
        valid = [i for i in range(6)]
        msg = [{"role": "system", "content": "You are an expert in analyzing the reason of choices and convert them into action in integer format which will be defined by users by calling the function named \"choose_act\". If the reason of choices inlcude a step by step actions plan, then you give a list of these corresponding integers in right order"}]
        msg.append({"role": "assistant", "content": "Sure, give me your reason of choices and I will convert it into corresponding integer based on user definition, and if there are more than one steps I will convert it into a integer list"})
        fuc_msg = utilities['fuc_msg']
        fuc = [
            {
                "name": "choose_act",
                "description":fuc_msg,
                "parameters":{
                    "type":"object",
                    "properties":{
                        "action":
                        {
                         "type":"integer",
                         "description":"the action to take ",
                        }  
                    },
                    "required":["action"]
                }
            }
        ]
        msg.append({"role": "user", "content": act_msg_s})
        write_log(args, save_path, f"Prompt message = \n\n{act_msg_s}")

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
                write_log(args, save_path, f"\nCaught an error: {e}\n")
                time.sleep(retry_delay)
                retry_delay *= 2  # double the delay each time we retry
        if isinstance(act, int):
            write_log(args, save_path, f"\n\n***************** Single Action *****************\n")
            write_log(args, save_path, f"{act_obj_pair[str(act)]}")
            return act_obj_pair[str(act)]
        elif isinstance(act, list):
            write_log(args, save_path, f"\n\n***************** Multiple Actions *****************\n\n")
            write_log(args, save_path, f"{act}")
        else:
            write_log(args, save_path, "act is neither an integer nor a list.")
        for i in range(len(act)):
            act[i] = act_obj_pair[str(act)]
    return act

# Get the observation representation description, to aid in the decision making
def get_desc(args, env_id, world_map, inv, act_his, obs, exp, pos_x, pos_y, arrow):
    desc = f"This is environment #{str(env_id)}\n"
    global save_path
    global gpt_map
    global utilities
    write_log(args, save_path, f"\n\n################## Start Describing ##################\n\n")
    
    if args.input:
        # A demo action when using input
        desc += "Description of observation representation"
        return desc
    else:
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
        act_his_s = ", ".join(act_his)
        desc_msg = utilities['desc_msg']
        arrow_s = arrow[0].lower() + arrow[1:]
        if args.goal:
            goal = f"Your goal is {obs['mission']}"
            desc_msg_s = desc_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, act_his_s, exp, goal, str(args.desc))
        else:
            desc_msg_s = desc_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, act_his_s, exp, "", str(args.desc))

        msg = [{"role": "system", "content": "Your mission is to understand deeply and follow a interpretation format to describe a text-based environment as follows:"}]
        msg.append({"role": "user", "content": sys_msg_s})
        msg.append({"role": "assistant", "content": "Sure, give me the real world map, inventory, past actions and experience, I will describe about it to aid in fulfilling the mission"})
        msg.append({"role": "user", "content": desc_msg_s})
        write_log(args, save_path, f"Prompt message = \n\n{desc_msg_s}")
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
        while True:
            try:
                rsp = openai.ChatCompletion.create(
                    model = gpt_map[args.gpt],
                    messages = msg,
                    temperature = args.temp,
                    max_tokens = args.desc
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
    
# Get the mapping list between 0,1,2,3 and environment names in a list
def get_env_id_mapping(args):
    global utilities
    id_mappings = []
    for env in utilities['env_id_maps'].strip().split("\n"):
        _, env_name = env.strip().split(", ")
        id_mappings.append(env_name)
    return id_mappings

# Getting the experience based on two observation, action chosen and action history
def get_exp(args, env_id, n_world_map, o_inv, act, o_obs, n_obs, o_world_map, n_inv, act_his, o_pos_x, o_pos_y, n_pos_x, n_pos_y, o_arrow, n_arrow):
    global save_path
    global gpt_map
    global utilities
    write_log(args, save_path, f"\n################## Starting Reflection ##################\n")
    if args.input:
        return "new experience"
    else:
        global sys_msg_s
        # Then we need the observation message, which we will fill the act_temp.txt
        refl_msg = utilities['refl_msg']
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
        refl_msg_s = refl_msg.format(str(env_id), str(o_pos_x), str(o_pos_y), str(o_arrow), str(o_world_map[env_id][0]).replace("'", ""), str(o_world_map[env_id][1]).replace("'", ""), str(o_world_map[env_id][2]).replace("'", ""), o_inv_s, act, o_goal, str(env_id), str(n_pos_x), str(n_pos_y), str(n_arrow), str(n_world_map[env_id][0]).replace("'", ""), str(n_world_map[env_id][1]).replace("'", ""), str(n_world_map[env_id][2]).replace("'", ""), n_inv_s, n_goal, act_his_s, str(max(int(args.lim * (env_id + 1) / len(args.envs)), args.lim // 2)))

        msg = [{"role": "system", "content": "Your mission is to discover the experience learned from exploring a text based environment, based on previous and after an action observation difference."}]
        msg.append({"role": "user", "content": sys_msg_s})
        msg.append({"role": "assistant", "content" : "Sure, give me two world map, inventory, past actions so I can discover experience from them"})
        write_log(args, save_path, f"\nPrompt Message = \n\n{refl_msg_s}")
        msg.append({"role": "user", "content": refl_msg_s})
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
        while True:
            try:
                rsp = openai.ChatCompletion.create(
                    model = gpt_map[args.gpt],
                    messages = msg,
                    temperature = args.temp, 
                    max_tokens = max(int(args.lim * (env_id + 1) / len(args.envs)), args.lim // 2)
                )
                n_exp = rsp["choices"][0]["message"]["content"]
                break
            except Exception as e:
                write_log(args, save_path, f"Caught an error: {e}\n")
                time.sleep(retry_delay)
                retry_delay *= 2  # double the delay each time we retry
        write_log(args, save_path, f"\n\n***************** Gained Experience *****************\n\n")
        write_log(args, save_path, f"{n_exp}")

        return n_exp

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

# Get the new inventory based on difference between o_obs and n_obs
def get_n_inv(args, n_obs, o_obs):
    n_obs_img_obj = n_obs['image'].transpose(1,0,2)[:, :, 0]
    o_obs_img_obj = o_obs['image'].transpose(1,0,2)[:, :, 0]
    indices = np.where(n_obs_img_obj != o_obs_img_obj)
    return o_obs_img_obj[indices[0][0]][indices[1][0]]


# Get the saving path for the current argument setting
def get_path(args):
    # Test if the model is getting directions from input
    if args.input:
        dir_n = "INPUT"
    else:
        dir_n = "GPT"

    timestamp = datetime.now().strftime(r"%Y-%m-%d %H-%M-%S")
    if args.all:
        env_names = "ALL"
    else:
        env_names = "_".join(args.envs)
    arg_list = ["seed", "gpt", "view", "goal", "static", "temp", "steps", "memo", "lim", "refresh", "desc", "reason"]
    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Combine them to create the full path
    full_path = os.path.join(dir_n, env_names, folder_name, str(timestamp))
    
    return full_path

# Function to get re-spawn position (when seed = 23 only)
def get_pos_m(args):
    global utilities
    key_name = "env_pos_" + str(args.seed)
    # parse the data and create matrices
    pos_m = {}
    for env in utilities[key_name].strip().split("\n"):
        env_id, x, y, arrow = env.strip().split(', ')
        pos_m[int(env_id)] = (int(x), int(y), arrow)
    return pos_m

def get_ratios(args, env_id, env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec):
    global save_path
    env_view_r = np.count_nonzero(env_view_rec[env_id]) / np.size(env_view_rec[env_id]) * 100
    env_view_r_s = "{:.3f}%".format(env_view_r)

    env_step_r = np.count_nonzero(env_step_rec[env_id]) / np.size(env_step_rec[env_id]) * 100
    env_step_r_s = "{:.3f}%".format(env_step_r)

    env_memo_r = np.count_nonzero(env_memo_rec[env_id]) / np.size(env_memo_rec[env_id]) * 100
    env_memo_r_s = "{:.3f}%".format(env_memo_r)

    obj_intr_r = np.count_nonzero(obj_intr_rec[env_id]) / np.size(obj_intr_rec[env_id]) * 100
    obj_intr_r_s = "{:.3f}%".format(obj_intr_r)

    obj_view_r = np.count_nonzero(obj_view_rec[env_id]) / np.size(obj_view_rec[env_id]) * 100
    obj_view_r_s = "{:.3f}%".format(obj_view_r)
    write_log(args, save_path, f"\nenv view ratio = {env_view_r_s}\nenv step ratio = {env_step_r_s}\nenv memo ratio = {env_memo_r_s}\nobj intr ratio = {obj_intr_r_s}\nobj view ratio = {obj_view_r_s}\n")
    
    return env_view_r, env_step_r, env_memo_r, obj_intr_r, obj_view_r

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
    global utilities
    # read data from txt file

    # parse the data and create matrices
    env_view_rec = {}
    env_step_rec = {}
    env_memo_rec = {}
    obj_intr_rec = {}
    obj_view_rec = {}

    for env in utilities['env_sizes'].strip().split("\n"):
        env_id, h, w = map(int, env.strip().split(','))
        env_view_rec[env_id] = np.zeros((h, w))
        env_step_rec[env_id] = np.zeros((h, w))
        env_memo_rec[env_id] = np.zeros((h, w))
        obj_intr_rec[env_id] = np.array([[0 for i in range(11)] for j in range(3)])
        obj_view_rec[env_id] = np.array([0 for i in range(11)])

    return env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec

# Get the reason of action
def get_reason(args, world_map, inv, act_his, obs, exp, desc, pos_x, pos_y, arrow, env_id):
    global save_path
    global gpt_map
    global sys_msg_s
    global utilities
    write_log(args, save_path, f"\n\n################## Start Deciding ##################\n\n")
    act_obj_pair = {"0": "left", "1": "right", "2": "toggle",
                    "3": "forward", "4": "pick up", "5": "drop off"}
    if args.input:
        # A demo action when using input
        reason = "Reason of choice"
        return reason, "action message"
    else:
        # Then we need the observation message
        reason_msg = utilities['reason_msg']
        obj_map_s = np.array2string(world_map[env_id][0]).replace("'", "").replace("\"","")
        col_map_s = np.array2string(world_map[env_id][1]).replace("'", "").replace("\"","")
        sta_map_s = np.array2string(world_map[env_id][2]).replace("'", "").replace("\"","")
        obj_idx = {0: "unseen", 1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava", 10: "agent"}
        if inv == 0:
            inv_s = f"You are not holding anything"
        else:
            inv_s = f"You are holding a {obj_idx[inv]}"
        act_his_s = ", ".join(act_his)
        arrow_s = arrow[0].lower() + arrow[1:]
        if args.goal:
            goal = f"Your goal is {obs['mission']}"
            reason_msg_s = reason_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, act_his_s, exp, goal, desc, str(args.reason))
        else:
            reason_msg_s = reason_msg.format(str(env_id), pos_x, pos_y, arrow_s, obj_map_s, col_map_s, sta_map_s, inv_s, act_his_s, exp, "", desc, str(args.reason))

        msg = [{"role": "system", "content":  "You mission to be an agent that's about to explore a text based world, with environment observation representation provided by user."}]
        msg.append({"role": "user", "content": sys_msg_s})
        msg.append({"role": "assistant", "content": "Sure, give me the real observation in world map, inventory, past actions and experience format and I will decide to make one move or multiple step moves."})
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
def get_world_maps(args):
    # parse the data and create world maps
    world_map = {}
    global utilities
    for env in utilities['env_sizes'].strip().split("\n"):
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

# Load the utilities JSON
def load_utilities_JSON(args,):
    with open(args.utilities, 'r', encoding='utf-8') as f:
        return json.load(f)

def log_act_records(args):
    global env
    global save_path
    global scn_table
    global env_table
    global obj_table
    global world_map_table
    global env_view_ratio
    global env_step_ratio
    global env_memo_ratio
    global obj_intr_ratio
    global n_exp
    global c_exp
    global env_view_rec
    global env_step_rec
    global env_memo_rec
    global world_map
    global env_id
    global reason
    global exp
    global desc
    global scn_table_df
    global env_table_df
    global obj_table_df
    global world_map_table_df
    global metrics_table_df

    img_array = env.render()
    img = Image.fromarray(img_array)
    img.save(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_act_{act}.png"))

    if args.wandb:
        scn_table.add_data(wandb.Image(img), reason_msg_s, reason, act, n_exp, c_exp)
        env_table.add_data(str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", ""))
        obj_table.add_data(str(obj_intr_rec[env_id]), str(obj_view_rec[env_id]))
        world_map_table.add_data(str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", ""))

    env_view_ratio, env_step_ratio, env_memo_ratio, obj_intr_ratio, obj_view_ratio = get_ratios(args, env_id, env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec)

    reason_length = count_tokens(reason)
    with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_reason_{str(reason_length)}.txt"), 'w') as file:
        file.write(reason)
    
    exp_length = count_tokens(exp)
    with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_exp_{str(exp_length)}.txt"), 'w') as file:
        file.write(c_exp)

    desc_length = count_tokens(desc)
    with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_desc_{str(desc_length)}.txt"), 'w') as file:
        file.write(desc)

    # Log the data to the dataframe
    scn_table_df.loc[len(scn_table_df)] = [str(img), reason_msg_s, reason, act, n_exp, c_exp]
    env_table_df.loc[len(env_table_df)] = [str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", "")]
    obj_table_df.loc[len(obj_table_df)] = [str(obj_intr_rec[env_id]), str(obj_view_rec[env_id])]
    world_map_table_df.loc[len(world_map_table_df)] = [str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", "")]
    metrics_table_df.loc[len(metrics_table_df)] = [env_view_ratio, env_memo_ratio, env_step_ratio, obj_view_ratio, obj_intr_ratio, exp_length]

    metrics = {
        f"env_{i}/env_view_ratio": env_view_ratio,
        f"env_{i}/env_memo_ratio": env_memo_ratio,
        f"env_{i}/env_step_ratio": env_step_ratio,
        f"env_{i}/obj_view_ratio": obj_view_ratio,
        f"env_{i}/obj_intr_ratio": obj_intr_ratio,
        f"env_{i}/exp_length": exp_length
    }

    if args.wandb:
        # Log the metrics
        wandb.log(metrics)

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

def sum_exp(args, n_exp, o_exp, act_his):
    global save_path
    global gpt_map
    global sys_msg_s
    write_log(args, save_path, f"\n\n################## Starting Summarizing ##################\n\n")
    if args.input:
        return "summarized experience"
    else:
        # Then we need the observation message, which we will fill the act_temp.txt
        act_his_s = ", ".join(act_his)
        sum_msg = utilities['sum_msg']
        sum_msg_s = sum_msg.format(o_exp, n_exp, act_his_s, str(max(int(args.lim * (env_id + 1) / len(args.envs)), args.lim // 2)))
        msg = [{"role": "system", "content": "You mission is comparing the new and past experience an agent learned from exploring a text-based environment as well as its past actions, analyze and then sum them up to a be more evolved, better experience to guide agent to explore better in the future"}]
        msg.append({"role": "assistant", "content": "Sure, give me your past and new experience, your past actions and I will summarize them for you."})
        write_log(args, save_path, f"Prompt Message = \n\n{sum_msg_s}")
        msg.append({"role": "user", "content": sum_msg_s})
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
        while True:
            try:
                rsp = openai.ChatCompletion.create(
                    model=gpt_map[args.gpt],
                    messages=msg,
                    temperature = args.temp, 
                    max_tokens = max(int(args.lim * (env_id + 1) / len(args.envs)), args.lim // 2)
                )
                c_exp = rsp["choices"][0]["message"]["content"]
                break
            except Exception as e:
                write_log(args, save_path, f"Caught an error: {e}\n")
                time.sleep(retry_delay)
                retry_delay *= 2  # double the delay each time we retry
        write_log(args, save_path, f"\n\n***************** Summarized Experience *****************\n\n")
        write_log(args, save_path, f"{c_exp}")
        return c_exp

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
        "--all",
        action = "store_true",
        help = "to load all the environments if given",
    )
    parser.add_argument(
        "--API-key",
        default = "./utilities/API/API_KEY",
        type = str,
        help = "the location to load your OpenAI API Key"
    )
    parser.add_argument(
        "--cross",
        action = "store_true",
        help = "whether will agent bring experience from the past environment"
    )
    parser.add_argument(
        "--desc",
        type = int,
        default = 50, 
        help = "the token limits for observation description"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "list of environment names, see the ./utilities/envs_mapping.txt for mapping between index and env",
        default = ["0"]
    )
    parser.add_argument(
        "--exp-src",
        type = str,
        help = "the starting experience read path"
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
        default = 50,
        help = "the tokens limit to the experience"
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
        "--reason",
        type = int,
        default = 50, 
        help = "the token limits for reason of choice"
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
        "--steps",
        type = int,
        default = 2000,
        help = "the maximum numbers of steps each environment will be taken"
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
    save_path = get_path(args)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Start running the specified environment(s), for each one, it has limited steps, whether it uses an existing
    # experience or an evolving experience will be dependent on the arguments. 
    write_log(args, save_path, f"################## Starting Experiment ##################\n")
    write_log(args, save_path, f"Configurations are:\n{args}\n")

    # The mapping for the GPT
    gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}

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
            name = datetime.now().strftime(r"Run %Y-%m-%d %H:%M:%S"),
            config = vars(args)
        )
    
    # get the environment list based on --all or --envs, if --all then replace args.envs as all environments
    if args.all:
        args.envs = [str(i) for i in range(58)]
    
    # Load the overall utitlies JSON
    utilities = load_utilities_JSON(args)

    # Get the observation map for all environments, with 3-dimension (object, color, status) and height, width.
    world_map = get_world_maps(args)
    
    # Get the two record matrix for all environments, with environment and object level
    env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec = get_rec(args)
    
    # The environment ID and enviornment name mapping list
    envs_id_mapping = get_env_id_mapping(args)
    
    # Get the position mapping for all environments, which include the x, y (in integer) and the direction Right string
    pos_m = get_pos_m(args)
    
    # Iterate over all the environment
    for i in args.envs:
        # We skip difficult environments first:
        if i in ["15", "16", "17", "18", "19", "20"]:
            continue
        # The i is a string representing environment ID, e.g. "1"
        env_id = int(i)
        
        sys_msg = utilities['sys_msg']
        sys_msg_s = sys_msg.format(str(world_map[env_id][0].shape[0]), str(world_map[env_id][0].shape[1]), str(args.memo))
        
        if args.goal:
            sys_msg_s += "\nYou will be prompted a goal specific to the environment.\n"

        write_log(args, save_path, f"\n################## System Message ##################\n")
        write_log(args, save_path, sys_msg_s)

        # For each new environment, the inventory is always 0
        inv = 0
        
        # For every new environment, the action history is always 0 (empty)
        act_his = []
        
        # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
        pos_x, pos_y, arrow = pos_m[env_id]
        
        # Get environment name from the mapping
        env_name = envs_id_mapping[env_id]

        write_log(args, save_path, f"Loading environment = {env_name}")

        env: MiniGridEnv = gym.make(
            id = env_name,
            render_mode = "rgb_array",
            agent_view_size = args.view,
            screen_size = args.screen
        )

        if args.wandb:
            scn_table = wandb.Table(columns = ["img", "text", "reason", "act", "n_exp", "c_exp"])
            env_table = wandb.Table(columns = ["env_view_rec", "env_step_rec", "env_memo_rec"])
            obj_table = wandb.Table(columns = ["obj_intr_rec", "obj_view_rec"])
            world_map_table = wandb.Table(columns = ["world_map_obj", "world_map_col", "world_map_sta"])
        
        # we save them to the csv
        # Define table structure
        scn_table_columns = ["Image", "Message", "Reason", "Action", "N_exp", "C_exp"]
        env_table_columns = ["Env_View", "Env_Step", "Env_Memo"]
        obj_table_columns = ["Obj_Intr", "Obj_View"]
        world_map_table_columns = ["World_Map_Object", "World_Map_Color", "World_Map_Status"]
        metrics_table_columns = ["env_view_ratio", "env_memo_ratio", "env_step_ratio", "obj_view_ratio", "obj_intr_ratio", "exp_length"]

        scn_table_df = pd.DataFrame(columns=scn_table_columns)
        env_table_df = pd.DataFrame(columns=env_table_columns)
        obj_table_df = pd.DataFrame(columns=obj_table_columns)
        world_map_table_df = pd.DataFrame(columns=world_map_table_columns)
        metrics_table_df = pd.DataFrame(columns=metrics_table_columns)

        # Initilize the environment
        obs, state = env.reset(seed=args.seed)

        # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
        p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
        
        # Iterate the agent exploration within the limit of args.steps
        if args.wandb:
            # scn_table.add_data(wandb.Image(img), act_msg_s, act, n_exp, c_exp)
            env_table.add_data(str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", ""))
            obj_table.add_data(str(obj_intr_rec[env_id]), str(obj_view_rec[env_id]))
            world_map_table.add_data(str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", ""))
        
        env_view_ratio, env_step_ratio, env_memo_ratio, obj_intr_ratio, obj_view_ratio = get_ratios(args, env_id, env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec)

        exp_length = count_tokens(exp)
        with open(os.path.join(save_path, f"env_{i}_idx_0_exp_{str(exp_length)}.txt"), 'w') as file:
            file.write(exp)

        with open(os.path.join(save_path, f"env_{i}_idx_0_reason_0.txt"), 'w') as file:
            file.write("Initial Reason of Choice")

        with open(os.path.join(save_path, f"env_{i}_idx_0_desc_0.txt"), 'w') as file:
            file.write("Initial Description")


        # Log the data to the dataframe
        env_table_df.loc[len(env_table_df)] = [str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", "")]
        obj_table_df.loc[len(obj_table_df)] = [str(obj_intr_rec[env_id]), str(obj_view_rec[env_id])]
        world_map_table_df.loc[len(world_map_table_df)] = [str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", "")]
        metrics_table_df.loc[len(metrics_table_df)] = [env_view_ratio, env_memo_ratio, env_step_ratio, obj_view_ratio, obj_intr_ratio, exp_length]

        metrics = {
            "env_view_ratio": env_view_ratio,
            "env_memo_ratio": env_memo_ratio,
            "env_step_ratio": env_step_ratio,
            "obj_view_ratio": obj_view_ratio,
            "obj_intr_ratio": obj_intr_ratio,
            "exp_length": exp_length
        }

        if args.wandb:
            # Log the metrics
            wandb.log(metrics)
            # wandb.log(metrics_table_df.iloc[-1].to_dict())

        img_array = env.render()
        img = Image.fromarray(img_array)
        img.save(os.path.join(save_path, f"env_{i}_action_0_start.png"))


        for j in range(args.steps):
            # We get a new action, during which update the record tables
            desc = get_desc(args, env_id, world_map, inv, act_his, obs, exp, pos_x, pos_y, arrow)
            reason, reason_msg_s = get_reason(args, world_map, inv, act_his, obs, exp, desc, pos_x, pos_y, arrow, env_id)
            act = get_action(args, reason)
            # We get the action from the act_hint, the act is a string format like "pick up"
            # With the new act, we convert it into the actions object
            if isinstance(act, str):
                execute_action(args, act)
                log_act_records(args)
            elif isinstance(act, list):
                for act_i in act:
                    if j < args.steps:
                        execute_action(args, act_i)
                        log_act_records(args)
                        j += 1
                    else:
                        break

        # Environment close() due to all steps finished      
        env.close()

        # Log datas to the wandb
        if args.wandb:
            wandb.log({f"Table/Screenshot for Environment #{i}": scn_table})
            wandb.log({f"Table/Environment Record for Environment #{i}": env_table})
            wandb.log({f"Table/Object Record for Environment #{i}": obj_table})
            wandb.log({f"Table/World Map for Environment #{i}": world_map_table})

        # Save to CSV
        scn_table_df.to_csv(os.path.join(save_path, f'scn_table_env_{i}.csv'), index=False)
        env_table_df.to_csv(os.path.join(save_path, f'env_table_env_{i}.csv'), index=False)
        obj_table_df.to_csv(os.path.join(save_path, f'obj_table_env_{i}.csv'), index=False)
        world_map_table_df.to_csv(os.path.join(save_path, f'world_map_table_env_{i}.csv'), index=False)
        metrics_table_df.to_csv(os.path.join(save_path, f'metrics_env_{i}.csv'), index=False)

    if args.wandb:
        wandb.finish()