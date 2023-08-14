import json
from utils.load_data import *

def fill_desc_user_0(env_id):
    desc_user_0 = train_msg['desc_user_0']
    choices = train_msg['choices']
    return desc_user_0.format(env_id, choices)

def fill_desc_user_1(args, env_id, pos_x, pos_y, direction, world_map, inv, past_actions, desc_lim):
    