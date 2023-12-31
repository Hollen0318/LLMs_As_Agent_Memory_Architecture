# Packages
import argparse
from classes.llm_agent import agent
from utils.gpt.chat import load_api_key

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action = "store_true",
        help = "to load all the environments if given",
    )
    parser.add_argument(
        "--API-KEY",
        default = "utils/api/API_KEY",
        type = str,
        help = "the location to load your OpenAI API Key"
    )
    parser.add_argument(
        "--bound",
        type = int,
        help = "the bound of memory both in forget and remember",
        default = 2
    )
    parser.add_argument(
        "--cross",
        action = "store_true",
        help = "whether will agent bring experience from the past environment or they will refresh their experiences once enter new environment during training"
    )
    parser.add_argument(
        "--delay",
        type = int,
        default = 5,
        help = "the number of seconds to delay when in OpenAI API Calling"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "list of environment names, see the data/input/env/env_ids.json for mapping between index and env",
        default = [0],
        type = int
    )
    parser.add_argument(
        "--eval",
        action = "store_true",
        help = "evaluation mode boolean, otherwise it is train mode"
    )
    parser.add_argument(
        "--exp-src",
        nargs = "+",
        type = str,
        help = "list of the starting experience read path, if it is less than the --envs, then we will use default one"
    )
    parser.add_argument(
        "--gpt",
        type = str,
        nargs = "+",
        choices = ["0", "1", "2", "3", "4", "5", "6", "7"],
        help = r'the version of gpt, type version number like 3 or 4, the correspondance relationship is gpt_map = {"0": "gpt-3.5-turbo", "1": "gpt-3.5-turbo-0301", "2": "gpt-3.5-turbo-0613", "3": "gpt-3.5-turbo-16k", "4": "gpt-3.5-turbo-16k-0613", "5": "gpt-4", "6": "gpt-4-0314", "7": "gpt-4-0613"} it should be the same length as env if we want to have different gpt for different envs',
        default = ["0"]
    )
    parser.add_argument(
        "--input",
        action = "store_true",
        help = "if the action and experience will be given by user instead of generating from GPT"
    )
    parser.add_argument(
        "--log",
        action = "store_true",
        help = "if to print the logging informations by print()"
    )
    parser.add_argument(
        "--neg-memo",
        action = "store_true",
        help = "will the agent have negative memory or only positive memory storage"
    )
    parser.add_argument(
        "--prj-name",
        type = str,
        help = "the project name for your wandb",
        default = "LLM As Agent"
    )
    parser.add_argument(
        "--retrain",
        action = "store_true",
        help = "retrain from the directory given in the retrain-src for number of steps defined in --steps"
    )
    parser.add_argument(
        "--retrain-src",
        type = str,
        help = "the retraining directory folder in string",
        default = r"output/seed_23/train/GPT/ENV 0 steps 200 gpt 0 temp 0.7 view 7/2023-08-20 23-23-41"
    )
    parser.add_argument(
        "--screen",
        type = int,
        default = 640,
        help = "set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--seed",
        help = "random seed for reproducing results, so far only supports 20, 21, 22, 23, 24",
        default = 23,
        type = int
    )
    parser.add_argument(
        "--steps",
        nargs = "+",
        help = "list of steps taken in the args.env, it should have the same length as the environment, used both in trainng and evaluation",
        default = [100],
        type = int
    )
    parser.add_argument(
        "--temp",
        nargs = "+",
        type = float,
        default = [0.7],
        help = "the list of temprature used by the OpenAI API, if it is less than the --envs, then we will use the first one throughout"
    )
    parser.add_argument(
        "--view",
        nargs = "+",
        type = int,
        default = [7],
        help = "set the number of grid spaces visible in agent-view, if it is less than the --envs, then we will use the first one throughout"
    )
    parser.add_argument(
        "--wandb",
        action = "store_true",
        help = "whether to use wandb to record experiments"
    )

args = parser.parse_args()
llm_agent = agent(args)

try:
    load_api_key(args.API_KEY)
    # Create the agent
    if args.eval:
        llm_agent.eval()
    else:
        if args.retrain:
            llm_agent.retrain()
        else:
            llm_agent.train()
    
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    for env_id in args.envs:
        llm_agent.save_table(env_id)  # This will always be executed
