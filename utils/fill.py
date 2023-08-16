import json
from utils.load_data import *
from utils.track import compose_world_map

def fill_desc_user_0(env_id):
    desc_user_0 = train_msg['desc_user_0']
    choices = train_msg['choices']
    return desc_user_0.format(env_id, choices)

def fill_desc_user_1(env_id, pos_x, pos_y, direction, world_map, inv, past_actions, desc_lim):
    desc_user_1 = train_msg['desc_user_1']
    c_world_map = str(compose_world_map(world_map))
    past_actions_s = ", ".join(past_actions) 
    return desc_user_1.format(str(env_id), str(pos_x), str(pos_y), str(direction), str(c_world_map), str(inv), past_actions_s, desc_lim)

def fill_reason_user_0(reason_lim):
    reason_user_0 = train_msg["reason_user_0"]
    return reason_user_0.format(reason_lim)