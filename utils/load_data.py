import json

# Load the utilities JSON
def load_json(address):
    with open(address, 'r', encoding='utf-8') as f:
        return json.load(f)
    
gpt_map = load_json(r"data/input/gpt_map.json")

env_ids = load_json(r"data/input/envs.json")

data = load_json(r"data/input/data.json")

goals = load_json(r"data/input/goals.json")

train_rec = load_json(r"data/input/train_rec.json")