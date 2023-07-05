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

# Function to return what GPT returns
def choose_act(action):
    return str(action)

# Describe the relative location based on the diff_x and diff_y, where the positve values
# means at the front or on the right by default
def describe_location(diff_x, diff_y):
    """Returns a description of a location based on the relative coordinates."""
    description = ""
    # Describe the front/back direction
    if diff_y >= 0:
        description += f"{abs(diff_y)} step(s) in front of you"
    elif diff_y < 0:
        description += f"{abs(diff_y)} step(s) in back of you"
    # Describe the left/right direction
    if diff_x > 0:
        description += f" and {abs(diff_x)} step(s) to your left"
    elif diff_x <= 0:
        description += f" and {abs(diff_x)} step(s) to your right"
    return description

# Convert the observation into environment description, it takes input of 
# surroundings, inventory and past experience at the hand
def obs_to_description(args, obs, inv, exp, env_id, act_his):
    if args.rel_des:
        dir = obs['direction']
        img = obs['image']
        goal = obs['mission'] if args.goal else None
        # The user location in the MiniGrid is always (view//2,view-1) where the x axis
        # points to the right and y axis points to the down direction 
        user_x, user_y = args.view // 2, args.view - 1 
        # We denote the direction based on the MiniGrid constant.py
        direction_dict = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}
        # Interpret each pixel in the observation
        description = f"This is Environment # {env_id}"
        descriptions = [f"You are facing {direction_dict[dir]}."]
        descriptions.append(description)
        # We need the front_object to be used for deterimine the inventory list,
        # for example, if we have an pickable object, we need to log it to the inventory list
        front_object = ""
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                # We load every squares in the image observation matrix with shape x,x,3
                # print(f"x = {x} y = {y} obs[{x},{y}] = {obs[x,y]} user_y = {user_y} user_x = {user_x}")
                
                # We want to skip the unseen parts
                if img[x,y][0] == 0:
                    continue

                # We will not record the observation for the agent itself to save the interpretation
                if y == user_y and x == user_x:
                    continue
                
                # We record the object in front of the agent for future use
                if y == user_y - 1 and x == user_x:
                    front_object = str(img[x,y])
                # Calculate the difference in coordinates
                diff_x, diff_y = user_x - x, user_y - y
                description = f"{img[x,y]} at {describe_location(diff_x, diff_y)}."
                descriptions.append(description)
        description = f"Your inventory status is {inv}"
        descriptions.append(description)
        descriptions_e = descriptions.copy()
        description = f"Your current experience is \n{exp}"
        descriptions_e.append(description)
        description = f"Front object is {front_object}"
        descriptions.append(description)
        descriptions_e.append(description)
        descriptions_e.append(f"Your past actions are:")
        description = ", ".join(act_his)
        descriptions_e.append(description)
        if goal is None:
            return "\n".join(descriptions), "\n".join(descriptions_e), front_object
        else:
            description = f"Your goal is \n{goal}"
            descriptions.append(description)
            descriptions_e.append(description)
            return "\n".join(descriptions), "\n".join(descriptions_e), front_object
    else:
        dir = obs['direction']
        img = obs['image'].transpose(1,0,2)
        goal = obs['mission'] if args.goal else None
        # The user location in the MiniGrid is always (view//2,view-1) where the x axis
        # points to the right and y axis points to the down direction 
        user_x, user_y = args.view - 1, args.view // 2
        # Change agent's location into [10,0,0]
        img[user_x, user_y] = [10,0,0]
        # We denote the direction based on the MiniGrid constant.py
        direction_dict = {0: 'east', 1: 'south', 2: 'west', 3: 'north'}
        # Interpret each pixel in the observation
        description = f"This is Environment # {env_id}"
        descriptions = [f"You are facing {direction_dict[dir]}."]
        descriptions.append(description)
        # We need the front_object to be used for deterimine the inventory list,
        # for example, if we have an pickable object, we need to log it to the inventory list
        front_object = ""
        description = f"Your observation is\n{img}"
        descriptions.append(description)
        for x in range(args.view):
            for y in range(args.view):
                # We record the object in front of the agent for future use
                if y == user_y and x == user_x - 1:
                    front_object = str(img[x,y])
        description = f"Your inventory status is {inv}"
        descriptions.append(description)
        descriptions_e = descriptions.copy()
        description = f"Your current experience is \n{exp}"
        descriptions_e.append(description)
        description = f"Front object is {front_object}"
        descriptions.append(description)
        descriptions_e.append(description)
        descriptions_e.append(f"Your past actions are:")
        description = ", ".join(act_his)
        descriptions_e.append(description)
        if goal is None:
            return "\n".join(descriptions), "\n".join(descriptions_e), front_object
        else:
            description = f"Your goal is \n{goal}"
            descriptions.append(description)
            descriptions_e.append(description)
            return "\n".join(descriptions), "\n".join(descriptions_e), front_object

# Get the mapping list between 0,1,2,3 and environment names in a list
def get_env_list():
    file_name = "./utilities/envs_mapping.txt"
    mappings = []
    with open(file_name, "r") as file:
        for line in file:
            _, env_name = line.strip().split(", ")
            mappings.append(env_name)
    return mappings

# Get the saving path for the current argument setting
def get_path(args):
    # Get today's date and format it as MM_DD_YYYY
    if args.all:
        env_names = "ALL"
    else:
        env_names = "_".join(args.envs)
    arg_list = ["seed", "gpt", "view", "input", "goal", "static", "temp", "steps", "all", "lim", "rel-des"]
    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Combine them to create the full path
    full_path = os.path.join(env_names, folder_name)
    return full_path

def get_action(args, text):
    if args.log:
        print(f"\n################## Starting Deciding ##################\n")
        print(f"Prompt Message =\n\n{text}\n")
    act_obj_pair = {"0": "left", "1": "right", "2": "toggle",
                    "3": "forward", "4": "pick up", "5": "drop off"}
    if args.input:
        if args.log:
            print(f"{open(args.sys_msg).read()}")
            print(f"{text}")
            print(f"{open(args.fuc_msg).read()}\n")
        valid = [str(i) for i in range(6)]
        while True:
            act = input("") 
            if act in valid:
                break
            else:
                continue
    else:
        valid = [i for i in range(6)]
        gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
        sys_msg = open(args.sys_msg).read()
        if args.goal:
            sys_msg += "\nYou will be prompted a goal specific to the environment.\n"
        msg = [{"role": "system", "content": sys_msg}]
        fuc_msg = open(args.fuc_msg).read()
        fuc = [{"name": "choose_act","description":fuc_msg,"parameters":{"type":"object", "properties":{"action":{"type":"integer", "description":"the action to take (in integer)","enum":[i for i in range(6)]}}}}]
        usr_msg = text + f"\n{fuc_msg}"      
        msg.append({"role": "user", "content": usr_msg})
        max_retries = args.max_rty  # maximum number of retries
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
    
        for attempt in range(max_retries):
            try:
                rsp = openai.ChatCompletion.create(
                model=gpt_map[args.gpt],
                messages=msg,
                functions = fuc,
                function_call = "auto",
                temperature = args.temp
                )
                rsp_msg = rsp["choices"][0]["message"]
                if rsp_msg.get("function_call"):
                    fuc_l = {
                        "choose_act": choose_act,
                    }
                    fuc_n = rsp_msg["function_call"]["name"]
                    if fuc_n == "choose_act":
                        fuc_c = fuc_l[fuc_n]
                        fuc_args = json.loads(rsp_msg["function_call"]["arguments"])
                        if fuc_args.get("action") in valid:
                            act = fuc_c(
                                action = fuc_args.get("action")
                            )
            except Exception as e:
                if args.log:
                    print(f"Caught an error: {e}")
                if attempt < max_retries - 1:  # no need to wait on the last attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # double the delay each time we retry
                else:
                    raise  # re-raise the last exception if all retries failed
    return act_obj_pair[act]

def get_exp(args, text, n_text, act, act_his, p_exp):
    if args.log:
        print(f"\n################## Starting Reflection ##################\n")
    if args.input:
        usr_msg = f"Old observation is:\n\n" + text 
        exp_msg = open(args.exp_msg).read()
        usr_msg += f"""\nYou have choose to do {act}\n\nNew observation is:\n{n_text}\n\nYour past actions are {", ".join(act_his)}\n\nYour past experience is {p_exp}\n\n{exp_msg}\n"""
        if args.log:
            print(f"Prompt Message = \n\n{usr_msg}")
        exp = input("Write your experience here")
    else:
        gpt_map = {"3":"gpt-3.5-turbo", "4":"gpt-4"}
        sys_msg = open(args.sys_msg).read()
        if args.goal:
            sys_msg += "You will be prompted a goal in the environment.\n"
        msg = [{"role": "system", "content": sys_msg}]
        usr_msg = f"Old observation is:\n\n" + text 
        exp_msg = open(args.exp_msg).read()
        usr_msg += f"""\nYou have choose to do {act}\n\nNew observation is:\n{n_text}\n\nYour past actions are {", ".join(act_his)}\n\nYour past experience is {p_exp}\n\n{exp_msg}\n"""
        if args.log:
            print(f"Prompt Message = \n\n{usr_msg}")
        msg.append({"role": "user", "content": usr_msg})
        max_retries = args.max_rty  # maximum number of retries
        retry_delay = args.rty_dly  # wait for 1 second before retrying initially
        for attempt in range(max_retries):
            try:
                rsp = openai.ChatCompletion.create(
                    model=gpt_map[args.gpt],
                    messages=msg,
                    temperature = args.temp, 
                    max_tokens = args.lim
                )
                exp = rsp["choices"][0]["message"]["content"]
            except:
                if attempt < max_retries - 1:  # no need to wait on the last attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # double the delay each time we retry
                else:
                    raise  # re-raise the last exception if all retries failed
    return exp

# Conver the text act into MiniGrid action object, update the inventory as well
def cvt_act(inv, act, fro_obj):
    act_obj_pair = {"left": Actions.left, "right": Actions.right, "toggle": Actions.toggle,
                        "forward": Actions.forward, "pick up": Actions.pickup, "drop off": Actions.drop}
    act_obj = act_obj_pair[act]
    l_fro_objs = fro_obj.strip("[]").split()
    l_fro_obj = [int(e) for e in l_fro_objs]
    if act == "pick up" and inv == 0 and l_fro_obj[0] in [5,6,7]:
        inv = l_fro_obj[0]
    elif act == "drop off" and l_fro_obj[0] == 1:
        inv = 0
    else:
        inv = inv
    return inv, act_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action = "store_true",
        help = "to load all the environments if given",
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
        "--exp-msg",
        type = str,
        default = "./utilities/exp_msg.txt",
        help = "message to hint for a experience"
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
        help = "the path to read the function helper message"
    )
    parser.add_argument(
        "--goal",
        action = "store_true",
        help = "include the text goal into the observation description"
    )
    parser.add_argument(
        "--gpt",
        type = str,
        choices = ["3", "4"],
        help = "the version of gpt, type version nuber only like 3.5 or 4",
        default = "3"
    )
    parser.add_argument(
        "--input",
        action = "store_true",
        help = "true if the action and experience will be given by user instead of GPT"
    )
    parser.add_argument(
        "--lim",
        type = int,
        default = 300,
        help = "the words limit to the experience"
    )
    parser.add_argument(
        "--log",
        action = "store_true",
        help = "print the logging informations by print()"
    )
    parser.add_argument(
        "--max-rty",
        type = int,
        default = 5,
        help = "the maximum number of delays in OpenAI API Calling"
    )
    parser.add_argument(
        "--overwrite",
        action = "store_true",
        help="overwrite the same experiment setting"
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "LLM As Agent"
    )
    parser.add_argument(
        "--rel-des",
        action = "store_true",
        help = "whether to use relative position description or pure array print as observation description" 
    )
    parser.add_argument(
        "--rty-dly",
        type = int,
        default = 1,
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
        default = 20,
        help = "the maximum numbers of steps each environment will be taken"
    )
    parser.add_argument(
        "--sys-msg",
        type = str,
        default = "./utilities/sys_msg.txt",
        help = "message to hint for an action"
    )
    parser.add_argument(
        "--temp",
        type = float,
        default = 0.0,
        help = "the temprature used by the OpenAI API"
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
    save_path = get_path(args)

    # Load the API key
    openai.api_key = open(args.API_key).read()
    if args.wandb:
        wandb.init(
            project = args.prj_name,
            name = datetime.now().strftime("Run %Y-%m-%d %H:%M:%S"),
            config = vars(args)
        )

    # Create or delte the save directory depending on the arguments
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if args.overwrite:
            for root, dirs, files in os.walk(save_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
        else:
            if args.log:                
                print(f"Same setting already exist, abort the experiment\nInclude the --overwrite to overwrite existing experiment")
            sys.exit(0)
    # Start running the specified environment(s), for each one, it has limited steps, whether it uses an existing
    # experience or an evolving experience will be dependant on the arguments. 
    envs_mapping = get_env_list()
    if args.log:
        print(f"\n################## Starting Experiment ##################\n")
        print(f"Configurations are:\n{args}")
    # Load the experience if it's given, and determine the training process based on --static
    if args.exp_src is not None:
        exp = open(args.exp_src).read()
    else:
        exp = ""
    
    act_idx = 0
    env_id = 0
    if args.all:
        args.envs = [str(i) for i in range(61)]
    if args.log:
        print(f"\n################## System Message ##################\n")
        print(f"{open(args.sys_msg).read()}")
    for i in args.envs:
        # For every new environment, the inventory is always 0 (empty)
        inv = 0
        env_name = envs_mapping[int(i)]
        if args.log:
            print(f"Loading environment = {env_name}")    
        env: MiniGridEnv = gym.make(
            id = env_name,
            render_mode = "human" if args.disp else "rgb_array",
            agent_view_size = args.view,
            screen_size = args.screen
        )
        if args.wandb:
            table = wandb.Table(columns=["img", "obs", "text", "act", "exp"])
        act_his = []
        obs, state = env.reset(seed=args.seed)
        for j in range(args.steps):
            # gain the text description and front object index
            text, text_e, fro_obj = obs_to_description(args, obs, inv, exp, env_id, act_his)
            # get an action in text e.g. forward, pick up
            act = get_action(args, text_e)
            if args.log:
                print(f"***************** Gained Action *****************\n")
                print(f"You have choose to do \"{act}\"")
            # using the action to determine the inventory and MiniGrid action object
            inv, act_obj = cvt_act(inv, act, fro_obj)
            if args.disp:
                scn = pyautogui.screenshot()
                scn.save(os.path.join(save_path, f"env_{i}_action_{act_idx}_{act}.png"))
            else:
                # make a screenshot
                img_array = env.render()
                img = Image.fromarray(img_array)
                img.save(os.path.join(save_path, f"env_{i}_action_{act_idx}_{act}.png"))
            # take the action returned either by api or user
            n_obs, reward, terminated, truncated, _= env.step(act_obj)
            n_text, ntext_e, n_fro_obj = obs_to_description(args, n_obs, inv, exp, env_id, act_his)
            # get a new experience
            if args.static:
                continue
            else:
                n_exp = get_exp(args, text, n_text, act, act_his, exp)
                with open(os.path.join(save_path, f"env_{i}_action_{act_idx}_{act}.txt"), "w") as f:
                    f.write(n_exp)
            act_his.append(act)
            if args.wandb:
            # log everything to the wandb    
                table.add_data(wandb.Image(img), obs, text_e, act, n_exp)
            obs = n_obs
            if args.log:
                print(f"\n***************** Gained Experience *****************n")
                print(f"{n_exp}")
            act_idx += 1
            exp = n_exp
        env_id += 1
        env.close()
        if args.wandb:
            wandb.log({f"Trajectory Table #{env_id} \"{env_name}\"":table})