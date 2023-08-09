# LLM As Agent

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
python train.py --log --steps 500 --desc 100 --lim 150 --memo 4 --reason 100 --wandb
```
The command python train.py --log --steps 500 --desc 100 --lim 150 --memo 4 --reason 100 --wandb runs the training script for an agent in a MiniGrid environment using OpenAI's GPT model. The script is configured to print logging information (--log), take a maximum of 500 steps in each environment (--steps 500), set a token limit of 100 for the observation description (--desc 100), set a token limit of 150 for the experience (--lim 150), allow the agent to remember past scenes for up to 4 steps (--memo 4), set a token limit of 100 for the reason behind each choice (--reason 100), and record the experiments using Weights & Biases (--wandb).

python train.py --log --steps 100 --desc 50 --lim 50 --memo 10 --reason 50 --wandb --envs 1
python train.py --log --steps 100 --desc 100 --lim 100 --memo 10 --reason 100 --wandb --envs 1
python train.py --log --steps 100 --desc 150 --lim 150 --memo 10 --reason 150 --wandb --envs 1 
python train.py --log --steps 100 --desc 200 --lim 200 --memo 10 --reason 200 --wandb --envs 1
python train.py --log --steps 100 --desc 250 --lim 250 --memo 10 --reason 250 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 300 --lim 300 --memo 10 --reason 300 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 350 --lim 350 --memo 10 --reason 350 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 400 --lim 400 --memo 10 --reason 400 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 450 --lim 450 --memo 10 --reason 450 --wandb --envs 1
python train.py --log --steps 100 --desc 500 --lim 500 --memo 10 --reason 500 --wandb --envs 1
python train.py --log --steps 100 --desc 550 --lim 550 --memo 10 --reason 550 --wandb --envs 1
python train.py --log --steps 100 --desc 600 --lim 600 --memo 10 --reason 600 --wandb --envs 1
python train.py --log --steps 100 --desc 650 --lim 650 --memo 10 --reason 650 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 700 --lim 700 --memo 10 --reason 700 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 750 --lim 750 --memo 10 --reason 750 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 800 --lim 800 --memo 10 --reason 800 --wandb --envs 1 --API-key ./utilities/API/API_KEY_YCQ
python train.py --log --steps 100 --desc 850 --lim 850 --memo 10 --reason 850 --wandb --envs 1
python train.py --log --steps 100 --desc 900 --lim 900 --memo 10 --reason 900 --wandb --envs 1
python train.py --log --steps 100 --desc 950 --lim 950 --memo 10 --reason 950 --wandb --envs 1
python train.py --log --steps 100 --desc 1000 --lim 1000 --memo 10 --reason 1000 --wandb --envs 1
