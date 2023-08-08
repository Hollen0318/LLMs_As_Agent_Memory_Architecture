# Packages
import pandas as pd
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
import numpy as np
import re

def get_goals_by_env_and_level(args, data, env_id, level_name):
    global save_path
    for environment in data["environments"]:
        if environment["id"] == env_id:
            env_name = environment["name"]
            for level in environment["level"]:
                if level["level"] == level_name:
                    return env_name, level["goals"]
    write_log(args, save_path, "goals not found")
    return None, None

# Get the saving path for the current argument setting
def get_path(args, env_id, lim):
    dir_n = "EVAL"
    timestamp = datetime.now().strftime(r"%Y-%m-%d %H-%M-%S")
    arg_list = ["seed", "gpt", "memo", "temp", "view"]
    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Combine them to create the full path
    full_path = os.path.join(dir_n, env_id, str(lim), folder_name, str(timestamp))
    
    return full_path

def load_dict_from_json(input_file):
    with open(input_file, 'r') as f:
        return json.load(f)

def locate_exp(args, x, y, gpt, memo, view, temp):
    global save_path
    base_dir = f"GPT/{x}/"
    base_pattern = f"desc_{y}_goal_False_gpt_{gpt}_lim_{y}_memo_{memo}_reason_{y}_seed_23_steps_"
    step_pattern = re.compile(f"steps_(\d+)_temp_{temp}_view_{view}")
    
    max_k = -1
    max_k_file_path = ""
    
    # Discover directories matching the base pattern
    steps_dirs = [d for d in os.listdir(base_dir) if d.startswith(base_pattern) and step_pattern.search(d)]
    for steps_dir in steps_dirs:
        # Get all date-named directories inside the steps_dir
        date_dirs = [d for d in os.listdir(os.path.join(base_dir, steps_dir)) if re.match(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", d)]
        for date_dir in date_dirs:
            # For each date directory, find the highest k value
            for file in os.listdir(os.path.join(base_dir, steps_dir, date_dir)):
                match = re.match(r"env_" + str(x) + r"_idx_(\d+)_exp_", file)
                if match:
                    k_val = int(match.group(1))
                    if k_val > max_k:
                        max_k = k_val
                        max_k_file_path = os.path.join(base_dir, steps_dir, date_dir, file)
    
    # Return content of the identified file
    if max_k_file_path:
        with open(max_k_file_path, 'r') as file:
            write_log(args, save_path, f"Loaded experience = {file.read()}\n\nThe path is {max_k_file_path}\n\nThe max_k experience length is {max_k}")
    else:
        write_log(args, save_path, "File not found.")


# Function to write the logging infos in to log save file
def write_log(args, save_path, text):
    if args.log:
        print(text)
    # Open the file in append mode
    with open(os.path.join(save_path, f"log.txt"), "a", encoding='utf-8') as file:
        # Write the strings to the file
        file.write(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--API-key",
        default = "./utilities/API/API_KEY",
        type = str,
        help = "the location to load your OpenAI API Key"
    )
    parser.add_argument(
        "--end",
        type = int,
        default = 100,
        help = "the end desc, lim, reason tokens"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "list of environment names like 0, 1, 2",
        default = ["0"]
    )
    parser.add_argument(
        "--eval",
        type = str,
        default = "./utlitilies/eval_envs.json",
        help = "the path to load your evaluation envs json"
    )
    parser.add_argument(
        "--goals",
        type = str,
        default = "./utilities/goals.json",
        help = "the path to load the goals json file"
    )
    parser.add_argument(
        "--gap",
        type = int, 
        help = "the incremental amount between the experience limit",
        default = 50
    )
    parser.add_argument(
        "--gpt",
        type = str,
        choices = ["3", "4"],
        help = "the version of gpt, type version number like 3 or 4",
        default = "3"
    )
    parser.add_argument(
        "--level",
        type = str,
        chocies = ["very easy", "easy", "medium", "hard", "very hard"]
    )
    parser.add_argument(
        "--log",
        action = "store_true",
        help = "print the logging informations by print()"
    )
    parser.add_argument(
        "--memo",
        type = int,
        help = "how long can agent remember past scenes",
        default = 10
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "LLM As Agent"
    )
    parser.add_argument(
        "--rty-dly",
        type = int,
        default = 5,
        help = "the number of seconds to delay when in OpenAI API Calling"
    )
    parser.add_argument(
        "--seed",
        type = int,
        help = "random seed for reproducing results",
        default = 23
    )
    parser.add_argument(
        "--start",
        type = int,
        default = 50,
        help = "the start desc, lim, reason tokens"
    )
    parser.add_argument(
        "--steps",
        type = int,
        default = 100,
        help = "the total number of steps agent allowed to complete the goal"
    )
    parser.add_argument(
        "--temp",
        type = float,
        default = 0.7,
        help = "the temprature used by the OpenAI API"
    )
    parser.add_argument(
        "--utilities",
        type = str,
        default = "utilities/utilities.json",
        help = "the path to load your utilities JSON file storing all texts, environment name, start position etc"
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

    args = parser.parse_args()

    # Dictionary of environment being evaluated
    eval_envs = load_dict_from_json(args.utilities)

    goals_dict = load_dict_from_json(args.goals)

    for env_id in args.envs:
        for lim in range(args.start, args.end + 1, args.gap):
            save_path = get_path(args, env_id, lim)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            exp = locate_exp(args, env_id, lim, args.gpt, args.memo, args.view, args.temp)
            if env_id == "0":
                env_name = goals_dict[env_id]["name"]
                goals = goals_dict[env_id]["Level"][args.level]

            elif env_id == "1":
                pass
            elif env_id == "2":
                pass
            elif env_id == "3":
                pass
            elif env_id == "4":
                pass
            elif env_id == "5":
                pass
            elif env_id == "6":
                pass
            elif env_id == "7":
                pass
            elif env_id == "8":
                pass
            elif env_id == "9":
                pass
            else:
                print("Invalid environment ID")

            for step in range(args.steps):
                