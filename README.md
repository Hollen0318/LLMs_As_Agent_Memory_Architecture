# LLM As Agent Training Script

## Description
This script allows the training of an agent using OpenAI's GPT model in a MiniGrid environment. The agent can interact with the environment, make decisions based on its observations, and log its actions and decisions for further analysis.

## Installation

### Prerequisites
- Python 3.x
- pip

### Packages Installation
```bash
pip install argparse openai gym wandb pandas PIL
```

## Usage

### Cloning the Repository
```bash
git clone https://github.com/general-robotics-duke/LLM_As_Agent.git
cd LLM_As_Agent
```

### Command Line Arguments

1. `--all`: Load all the environments.
2. `--API-key`: Location to load your OpenAI API Key. Default is `./utilities/API/API_KEY`.
3. `--cross`: Whether the agent will bring experience from the past environment.
4. `--desc`: Token limits for observation description. Default is `50`.
5. `--envs`: List of environment names. Default is `["0"]`. See `./utilities/envs_mapping.txt` for mapping between index and environment.
6. `--exp-src`: Starting experience read path.
7. `--goal`: Include the text goal into the observation description.
8. `--gpt`: The version of GPT, either `3` or `4`. Default is `3`.
9. `--input`: If the action and experience will be given by the user instead of generating from GPT.
10. `--lim`: Token limit for the experience. Default is `50`.
11. `--log`: Print logging information.
12. `--memo`: How long the agent can remember past scenes. Default is `5`.
13. `--prj-name`: The project name for your wandb. Default is `LLM As Agent`.
14. `--reason`: Token limit for the reason of choice. Default is `50`.
15. `--rty-dly`: Delay in seconds during OpenAI API Calling. Default is `5`.
16. `--screen`: Resolution for pygame rendering. Default is `640`.
17. `--seed`: Random seed for reproducing results. Default is `23`.
18. `--steps`: Maximum number of steps each environment will be taken. Default is `2000`.
19. `--temp`: Temperature used by the OpenAI API. Default is `0.7`.
20. `--utilities`: Path to load the utilities JSON file. Default is `utilities/utilities.json`.
21. `--view`: Number of grid spaces visible in agent-view. Default is `7`.
22. `--wandb`: Use wandb to record experiments.

### Demo Usage
```bash
python train.py --log --steps 5 --desc 40 --lim 40 --memo 5 --reason 40
```
