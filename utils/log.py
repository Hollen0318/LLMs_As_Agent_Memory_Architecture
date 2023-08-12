import os
from datetime import datetime

# Get the saving path for the current argument setting
def get_path(args):
    # Test if the agent is controlled by input (human) or from LLMs
    if args.input:
        dir_n = "INPUT"
    else:
        dir_n = "GPT"
    # For same settings, there may be multiple experiment so it's important to distinguish time
    timestamp = datetime.now().strftime(r"%Y-%m-%d %H-%M-%S")

    # If the exprinment includes all the environments, then we use 'ALL"
    if args.all:
        env_names = "ALL"
    else:
        env_names = "_".join(args.envs)

    # These are the parmaters may change in the experiment, they are for us to distinguish them
    arg_list = ["seed", "gpt", "view", "goal", "temp", "steps", "memo", "lim", "desc", "reason"]

    # Create a folder name from the argument parser args
    folder_name = '_'.join(f'{k}_{v}' for k, v in vars(args).items() if k in arg_list)
    # Combine them to create the full path
    full_path = os.path.join(dir_n, env_names, folder_name, str(timestamp))
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


# Function to write the logging infos in to log save file
def write_log(args, save_path, text):
    if args.log:
        print(text)
    # Open the file in append mode
    with open(os.path.join(save_path, f"log.txt"), "a", encoding='utf-8') as file:
        # Write the strings to the file
        file.write(text)