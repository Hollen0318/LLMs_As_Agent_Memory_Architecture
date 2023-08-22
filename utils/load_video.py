import os
import json
import pandas as pd
# Load the utilities JSON
def load_json(address):
    with open(address, 'r', encoding='utf-8') as f:
        return json.load(f)

font_size = load_json(r"data/input/video/font_size.json")
content = load_json(r"data/input/video/content.json")
pos = load_json(r"data/input/video/pos.json")
img_size = load_json(r"data/input/video/img_size.json")
line_size = load_json(r"data/input/video/line_size.json")

def load_tables(load_address, env_id):
    rec_table_n = f"rec_table_env_{env_id}.csv"
    scn_table_n = f"scn_table_env_{env_id}.csv"
    world_map_table_n = f"world_map_table_env_{env_id}.csv"
    length_table_n = f"length_table_env_{env_id}.csv"
    metrics_table_n = f"metrics_table_env_{env_id}.csv"
    rec_table = pd.read_csv(os.path.join(load_address, rec_table_n))
    scn_table = pd.read_csv(os.path.join(load_address, scn_table_n))
    world_map_table = pd.read_csv(os.path.join(load_address, world_map_table_n))
    length_table = pd.read_csv(os.path.join(load_address, length_table_n))
    metrics_table = pd.read_csv(os.path.join(load_address, metrics_table_n))
    return rec_table, scn_table, world_map_table, length_table, metrics_table