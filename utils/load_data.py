import json
import pandas as pd
from classes.Minigrid.minigrid.core.actions import Actions
import os
import numpy as np
import re

def extract_actions(images_data):
    return [image_info['act'] for image_info in images_data]
    
def get_pos_m(env_pos):
    pos_m = {}
    for env in env_pos.strip().split("\n"):
        env_id, x, y, direction = env.strip().split(', ')
        pos_m[env_id] = (int(x), int(y), direction)
    return pos_m

def load_images_info(directory_path):
    images_info = []
    
    for filename in os.listdir(directory_path):
        if not filename.endswith(".png"):
            continue
        
        # Updated regex pattern to handle spaces in 'act'
        match = re.match(r"env_(\d+)_(\d+)_(\w+\s*\w*)_x_(\d+)_y_(\d+)_d_([\w_]+)\.png", filename)
        
        if match:
            env_id = int(match.group(1))
            image_id = int(match.group(2))
            act = match.group(3)   # This will now capture "pick up" and "drop off" as well
            x = int(match.group(4))
            y = int(match.group(5))
            direction = match.group(6).replace('_', '/')
            
            image_info = {
                'env_id': env_id,
                'image_id': image_id,
                'act': act,
                'x': x,
                'y': y,
                'direction': direction,
                'filename': filename
            }
            
            images_info.append(image_info)
    
    return sorted(images_info, key=lambda x: x['image_id'])

# Load the utilities JSON
def load_json(address):
    with open(address, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_retrain(retrain_src):
    images_data = load_images_info(retrain_src)
    env_id = images_data[0]["env_id"]
    pos_x, pos_y, direction = images_data[-1]["x"], images_data[-1]["y"], images_data[-1]["direction"]
    length_table_n = f"length_table_env_{env_id}.csv"
    metrics_table_n = f"metrics_table_env_{env_id}.csv"
    rec_table_n = f"rec_table_env_{env_id}.csv"
    scn_table_n = f"scn_table_env_{env_id}.csv"
    world_map_table_n = f"world_map_table_env_{env_id}.csv"
    length_table = pd.read_csv(length_table_n)
    metrics_table = pd.read_csv(metrics_table_n)
    rec_table = pd.read_csv(rec_table_n)
    scn_table = pd.read_csv(scn_table_n)
    world_map_table = pd.read_csv(world_map_table_n)
    env_view = string_to_numpy_array(rec_table.iloc[-1]["env_view"])
    env_step = string_to_numpy_array(rec_table.iloc[-1]["env_step"])
    env_memo = string_to_numpy_array(rec_table.iloc[-1]["env_memo"])
    obj_view = string_to_numpy_array(rec_table.iloc[-1]["obj_view"])
    toggle = string_to_numpy_array(rec_table.iloc[-1]["toggle"])
    pickup = string_to_numpy_array(rec_table.iloc[-1]["pick up"])
    dropoff = string_to_numpy_array(rec_table.iloc[-1]["dropoff"])
    h, w = env_view.shape[0], env_view.shape[1]
    world_map = np.empty((3, h, w), dtype = object)    
    world_map[0] = string_to_object_array(world_map_table.iloc[-1]["world_map_obj"])
    world_map[1] = string_to_object_array(world_map_table.iloc[-1]["world_map_col"])
    world_map[2] = string_to_object_array(world_map_table.iloc[-1]["world_map_sta"])
    obj_intr = {}
    obj_intr["toggle"] = toggle
    obj_intr["drop off"] = dropoff
    obj_intr["pick up"] = pickup
    rec = {}
    rec["env_view"], rec["env_step"], rec["env_memo"], rec["obj_intr"], rec["obj_view"] = env_view, env_step, env_memo, obj_intr, obj_view
    exp = scn_table.iloc[-1]["s_exp"]
    act_list = extract_actions(images_data)
    return env_id, world_map, rec, exp, pos_x, pos_y, direction, act_list

def string_to_numpy_array(data):
    # Remove outer brackets and split by spaces and newlines.
    numbers = [float(num) if '\n' in data else int(num) for num in data.replace('[', '').replace(']', '').split() if num.strip()]

    # Create numpy array
    array = np.array(numbers, dtype=np.float64 if '\n' in data else int)

    # Reshape if the original string is 2D
    if '\n' in data:
        # Get the number of rows by counting newline characters and adding 1.
        rows = data.count('\n') + 1
        # Reshape the array
        array = array.reshape(rows, -1)

    return array

def string_to_object_array(data):
    # Split the data string by newlines
    lines = data.split('\n')
    
    # For each line, strip whitespace, remove outer square brackets, 
    # split by spaces, and keep the value as is.
    rows = [line.strip().replace('[', '').replace(']', '').split() for line in lines]
    
    # Convert to a numpy array with dtype 'object'
    array = np.array(rows, dtype=object)
    
    return array

gpt_map = load_json(r"data/input/gpt/gpt_map.json")

env_ids = load_json(r"data/input/env/env_ids.json")

env_pos = load_json(r"data/input/env/env_pos.json")

env_sizes = load_json(r"data/input/env/env_sizes.json")

goals = load_json(r"data/input/goals/goals.json")

minigrid_mission = load_json(r"data/input/goals/minigrid_mission.json")

train_msg = load_json(r"data/input/msg/train_msg.json")

rpt_msg = load_json(r"data/input/msg/rpt_msg.json")

eval_msg = load_json(r"data/input/msg/eval_msg.json")

lim = load_json(r"data/input/lim/lim.json")

obs_rep = load_json(r"data/input/obs/obs_rep.json")

int_act = load_json(r"data/input/act/int_act.json")