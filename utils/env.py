import gymnasium as gym
from classes.Minigrid.minigrid.core.actions import Actions
from classes.Minigrid.minigrid.minigrid_env import MiniGridEnv
from utils.load_data import *

def start_env(args, env_id):
    env: MiniGridEnv = gym.make(
        id = env_ids[str(env_id)],
        render_mode = "rgb_array",
        agent_view_size = args.view[env_id],
        screen_size = args.screen
    )
    return env

def reset_env(env, seed):
    return env.reset(seed=seed)