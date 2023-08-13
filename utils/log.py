import os
from datetime import datetime

# Get the saving path for the current argument setting
def get_path(args, seed, steps, env_id):
    # The experiment should be classified first in seed, 
    # then mode (Train and Evaluation), then depends on GPT or INPUT,
    # then environment id, depends on its cross property,
    # we create single folder for each environment or together
    # then inside we use parameter setting as folder name
    # then we use the datetime
    seed_dir = "seed_" + str(seed)
    if args.eval:
        mode_dir = "eval"
    else:
        mode_dir = "train"
    # Test if the agent is controlled by input (human) or from LLMs
    if args.input:
        output_dir = "INPUT"
    else:
        output_dir = "GPT"
    if args.cross:
        if args.all:
            env_dir = "ALL"
        else:
            env_dir = "_".join(args.envs)
        steps_s = "_".join(args.steps)
        env_dir = "cross " + env_dir + f" steps {steps_s}"
    else:
        env_dir = str(env_id) + f" steps {steps}"
    # For same settings, there may be multiple experiment so it's important to distinguish time
    timestamp = datetime.now().strftime(r"%Y-%m-%d %H-%M-%S")
    # These are the parmaters may change in the experiment, they are for us to distinguish them
    arg_list = ["gpt", "view", "temp", "memo", "lim", "desc", "reason"]

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