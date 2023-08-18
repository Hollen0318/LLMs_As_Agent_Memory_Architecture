import json
from utils.load_data import *
from utils.track import compose_world_map

def fill_desc_user_1(env_id, pos_x, pos_y, direction, world_map, inv, past_actions, exp, desc_lim):
    desc_user_1 = train_msg['desc_user_1']
    c_world_map = compose_world_map(world_map)
    past_actions_s = ", ".join(past_actions) 
    return desc_user_1.format(str(env_id), str(pos_x), str(pos_y), str(direction), c_world_map, inv_s(inv), past_actions_s, exp, str(desc_lim))

def fill_reason_user_0(reason_lim):
    reason_user_0 = train_msg["reason_user_0"]
    return reason_user_0.format(str(reason_lim))

def fill_n_exp_user_0(act_l, env_id, pos_x, pos_y, direction, world_map, inv, past_actions, n_exp_lim):
    n_exp_user_0 = train_msg["n_exp_user_0"]
    c_world_map = compose_world_map(world_map)
    past_actions_s = ", ".join(past_actions) 
    act_l_s = ", ".join(act_l)
    return n_exp_user_0.format(act_l_s, str(env_id), str(pos_x), str(pos_y), str(direction), c_world_map, inv_s(inv), past_actions_s, str(n_exp_lim))

def fill_s_exp_user_0(s_exp_lim):
    s_exp_user_0 = train_msg["s_exp_user_0"]
    return s_exp_user_0.format(str(s_exp_lim))

def inv_s(inv):
    return f"You are holding {obs_rep['object'][str(inv)]}"