# Packages
import argparse
from classes.llm_agent import agent
from utils.gpt.initialize_gpt import load_api_key


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action = "store_true",
        help = "to load all the environments if given",
    )
    parser.add_argument(
        "--API-key",
        default = "./utilities/API/API_KEY",
        type = str,
        help = "the location to load your OpenAI API Key"
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
        help = "list of environment names, see the ./utilities/envs_mapping.txt for mapping between index and env",
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
        help = "the starting experience read path",
        default = ["start_exp.txt"]
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
        help = "print the logging informations by print()"
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
        "--screen",
        type = int,
        default = 640,
        help = "set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--seed",
        help = "random seed for reproducing results",
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
        help = "the list of temprature used by the OpenAI API"
    )
    parser.add_argument(
        "--view",
        nargs = "+",
        type = int,
        default = [7],
        help = "set the number of grid spaces visible in agent-view "
    )
    parser.add_argument(
        "--wandb",
        action = "store_true",
        help = "whether to use wandb to record experiments"
    )
    args = parser.parse_args()
    load_api_key(args.API_KEY)
    # Create the agent
    llm_agent = agent(args)
    llm_agent.train()