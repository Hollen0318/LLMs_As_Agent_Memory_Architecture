# Packages
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
import sys
import pyautogui
import numpy as np

# Function to return what GPT returns in sring format
def choose_act(action):
    return str(action)

# Get the saving path for the current argument setting
def get_path(args):
    # Test if the model is getting directions from input
    if args.input:
        dir_n = "INPUT"
    else:
        dir_n = "GPT"
    # Get today's date and format it as MM_DD_YYYY
    if args.all:
        env_names = "ALL"
    else:
        env_names = "_".join(args.envs)
    arg_list = ["seed", "gpt", "view", "goal", "static", "temp", "steps", "lim", "refresh"]
    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Create current timestamp to store files
    timestamp = datetime.now().strftime("Run %Y-%m-%d %H:%M:%S"),
    # Combine them to create the full path
    full_path = os.path.join(dir_n, env_names, timestamp, folder_name)
    return full_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action = "store_true",
        help = "to load all the environments if given",
    )
    parser.add_argument(
        "--act-temp",
        default = "./utilities/act_temp.txt",
        type = str,
        help = "the location to load your action prompt message template"
    )
    parser.add_argument(
        "--API-key",
        default = "./utilities/API/API_1",
        type = str,
        help = "the location to load your OpenAI API Key"
    )
    parser.add_argument(
        "--disp",
        action = "store_true",
        help = "display the environment rendering (human) if in GUI environment"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "list of environment names, see the ./utilities/envs_mapping.txt for mapping between index and env",
        default = ["1"]
    )
    parser.add_argument(
        "--env-id-maps",
        type = str,
        help = "the environment ID and environment name mapping",
        default = "./utilities/env_id_maps.txt"
    )
    parser.add_argument(
        "--env-sizes",
        type = str,
        help = "the environment ID and environment map size",
        default = "./utilities/env_sizes.txt"
    )
    parser.add_argument(
        "--env-pos",
        type = str,
        help = "the re-spawn position of the agent in each map (when seed = 23)",
        default = "./utilities/env_pos.txt"
    )
    parser.add_argument(
        "--exp-src",
        type = str,
        help = "the starting experience read path"
    )
    parser.add_argument(
        "--fuc-msg",
        type = str,
        default = "./utilities/fuc_msg.txt",
        help = "the path to read the function prompt message"
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
        default = 1000,
        help = "the words limit to the experience"
    )
    parser.add_argument(
        "--log",
        action = "store_true",
        help = "print the logging informations by print()"
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "LLM As Agent"
    )
    parser.add_argument(
        "--refl-temp",
        default = "./utilities/refl_temp.txt",
        type = str,
        help = "the location to load your reflect prompt message template"
    )
    parser.add_argument(
        "--rpt-temp",
        default = "./utilities/rpt_temp.txt",
        type = str,
        help = "the location to load your exploration report template format"
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
        "--static",
        action = "store_true",
        help = "the agent will not update experience during the exploration"
    )
    parser.add_argument(
        "--steps",
        type = int,
        default = 2000,
        help = "the maximum numbers of steps each environment will be taken"
    )
    parser.add_argument(
        "--sum-temp",
        default = "./utilities/sum_temp.txt",
        type = str,
        help = "the location to load your summarization prompt message template"
    )
    parser.add_argument(
        "--sys-msg",
        type = str,
        default = "./utilities/sys_msg.txt",
        help = "the location to load your hint message as agent's system background"
    )
    parser.add_argument(
        "--temp",
        type = float,
        default = 0.7,
        help = "the temprature used by the OpenAI API"
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
    parser.add_argument(
        "--refresh",
        type = int,
        default = 20,
        help = "the longest number of actionn historys"
    )
    parser.add_argument(
        "--forget-t",
        type = int,
        default = 10,
        help = "for how long will the agent starts to forget regions outside of view"
    )
    parser.add_argument(
        "--forget-r",
        type = float,
        default = 0.8,
        help = "for how likely the agent will forget regions esceeding the forget-t"
    )

