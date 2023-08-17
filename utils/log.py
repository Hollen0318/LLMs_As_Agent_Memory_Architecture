import os
from datetime import datetime
from utils.load_data import lim

# Get the saving path for the current argument setting
def get_path(args, env_id):
    # The experiment should be classified first in seed, 
    # then mode (Train and Evaluation), then depends on GPT or INPUT,
    # then environment id, depends on its cross property,
    # we create single folder for each environment or together
    # then inside we use parameter setting as folder name
    # then we use the datetime
    root_dir = "data/output"
    seed_dir = "seed_" + str(args.seed)
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
            env_dir = "ALL ENV"
        else:
            env_dir = "ENV " + "_".join(args.envs)
        steps_s = "_".join(args.steps)
        env_dir = "cross " + env_dir + f" steps {steps_s}" + f" gpt {args.gpt[0]} temp {args.temp[0]} view {args.view[0]}"
    else:
        env_dir = "ENV " + str(env_id) + f" steps {args.steps[env_id]} gpt {args.gpt[env_id]} temp {args.temp[env_id]} view {args.view[env_id]}"
    # For same settings, there may be multiple experiment so it's important to distinguish time
    timestamp = datetime.now().strftime(r"%Y-%m-%d %H-%M-%S")
    # These are the parmaters may change in the experiment, they are for us to distinguish them
    # Combine them to create the full path
    full_path = os.path.join(root_dir, seed_dir, mode_dir, output_dir, env_dir, str(timestamp))
    
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

        
# reason_length = count_tokens(reason)
# with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_reason_{str(reason_length)}.txt"), 'w') as file:
#     file.write(reason)

# exp_length = count_tokens(exp)
# with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_exp_{str(exp_length)}.txt"), 'w') as file:
#     file.write(c_exp)

# desc_length = count_tokens(desc)
# with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_desc_{str(desc_length)}.txt"), 'w') as file:
#     file.write(desc)