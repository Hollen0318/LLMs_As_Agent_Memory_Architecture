import json

# Load the utilities JSON
def load_json(address):
    with open(address, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# Function to get re-spawn position (when seed = 23 only)
def get_pos_m(env_pos):
    pos_m = {}
    for env in env_pos.strip().split("\n"):
        env_id, x, y, arrow = env.strip().split(', ')
        pos_m[env_id] = (int(x), int(y), arrow)
    return pos_m
    
gpt_map = load_json(r"data/input/gpt_map.json")

env_ids = load_json(r"data/input/env_ids.json")

env_pos = load_json(r"data/input/env_pos.json")

env_sizes = load_json(r"data/input/env_sizes.json")

goals = load_json(r"data/input/goals.json")

minigrid_mission = load_json(r"data/input/minigrid_mission.json")

train_msg = load_json(r"data/input/train_msg.json")

train_rec = load_json(r"data/input/train_rec.json")

eval_msg = load_json(r"data/input/eval_msg.json")

eval_rec = load_json(r"data/input/eval_rec.json")