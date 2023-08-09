#!/usr/bin/env python3

from __future__ import annotations
import os
import gymnasium as gym
import pygame
from gymnasium import Env
import numpy as np
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
import json

# Get the mapping list between 0,1,2,3 and environment names in a list
def get_env_id_mapping(args):
    global utilities
    id_mappings = []
    for env in utilities['env_id_maps'].strip().split("\n"):
        _, env_name = env.strip().split(", ")
        id_mappings.append(env_name)
    return id_mappings

# Load the utilities JSON
def load_utilities_JSON(args,):
    with open(args.utilities, 'r', encoding='utf-8') as f:
        return json.load(f)

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
        index = None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.index = index
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        print(f"obs img = {str(obs['image'][:, :, 0])} step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            img_array = self.env.render()
            img = Image.fromarray(img_array)
            img.save(os.path.join(args.save, f"env_{self.index}.png"))
            self.env.close()
            self.closed = True
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse
    import wandb
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--all",
        action = "store_true",
        help = "to load all the environments if given",
    )
    parser.add_argument(
        "--env-id-maps",
        type = str,
        help = "the environment ID and environment name mapping",
        default = r"../../utilities/env_id_maps.txt"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "list of environment names, see the ./utilities/envs_mapping.txt for mapping between index and env",
        default = ["1"]
    )
    parser.add_argument(
        "--rgb-view",
        action = "store_true",
        help = "whether to show the environment observation by RGB array"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=r"C:\Users\holle\OneDrive - Duke University\Research\LLM_As_Agent\utilities\env_scn_24",
        help="the save place for the screenshot"
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--tile-size", 
        type=int, 
        help="size at which to render tiles", 
        default=32
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "Minigrid Manaual Control As Agent"
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
        "--utilities",
        type = str,
        default = r"C:\Users\holle\OneDrive - Duke University\Research\LLM_As_Agent\utilities\utilities.json",
        help = "the path to load your utilities JSON file storing all texts, environment name, start position etc"
    )
    parser.add_argument(
        "--view",
        type = int,
        default = 3,
        help = "set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--wandb",
        action = "store_true",
        help = "whether to use wandb to record experiments"
    )
    args = parser.parse_args()
    
    # Load the overall utitlies JSON
    utilities = load_utilities_JSON(args)

    if args.wandb:
        wandb.init(
            project = args.prj_name,
            name = datetime.now().strftime("Run %Y-%m-%d %H:%M:%S"),
            config = vars(args)
        )

    envs_id_mapping = get_env_id_mapping(args)

    # get the environment list based on --all or --envs, if --all then replace args.envs as all environments
    if args.all:
        args.envs = [str(i) for i in range(57)]

    for i in args.envs:
        print(f"loading environment #{i}")
        env_id = int(i)
        # For each new environment, the inventory is always 0
        inv = 0
        # For every new environment, the action history is always 0 (empty)
        act_his = []
        # Get environment name from the mapping
        env_name = envs_id_mapping[env_id]

        env: MiniGridEnv = gym.make(
            id = env_name,
            render_mode = "rgb_array",
            # render_mode = "human",
            agent_view_size = args.view,
            screen_size = args.screen
        )
        # env = RGBImgPartialObsWrapper(env, 32)
        # manual_control = ManualControl(env, seed=args.seed, index = i)
        # manual_control.start()
        # Initilize the environment
        obs, state = env.reset(seed=args.seed)
        img_array = env.render()
        img = Image.fromarray(img_array)
        img.save(os.path.join(args.save, f"env_{i}.png"))
        print(f"obs = {str(obs['mission'])}")
        env.close()