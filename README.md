# LLM As Agent

Project Repository for LLM As Agent project code

## Introduction

This repository includes algorithm for training an agent embodied with LLM to explore environments from scratch, adapt during exploration.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have `git` and `conda` installed on your machine. To install these, you can use the following commands:

### Installing dependencies

```shell
git clone https://github.com/general-robotics-duke/LLM_As_Agent.git
cd LLMS_AS_AGENT
conda env create -f environment.yml
conda activate minigrid
```

## Quick Start

0. Paste your OpenAI API Key under the utilities/API/API_1 if you don't want to specify it in the run. 

1. Test the environment with `--API-key` stored under `./utilities/API/API_1`,`--sys-msg` under `./utilities/n_sys_msg.txt` with all other settings in default. (`--overwrite` will replace the same experiment configs experiment setting)

```shell
python start.py --API-key ./utilities/API/API_1 --envs 0 --log --mry --overwrite --sys-msg ./utilities/n_sys_msg.txt --wandb
```

2. Test the environment with `--API-key` stored under `./utilities/API/API_1`,`--sys-msg` under `./utilities/n_sys_msg.txt`, number of steps as 50, exploring all the environment and limiting the experience tokens limit to 1000 with all other settings in default. 
```shell
python start.py --all --API-key utilities/API/API_1 --lim 1000 --log --mry --overwrite --steps 50 --sys-msg ./utilities/n_sys_msg.txt --view 5 --wandb 
```

The screenshots, observation, experience and actions being taken will be uploaded to the wandb as well as saved under environmentID/configs. 
For example:
`ALL\all_True_goal_False_gpt_3_input_False_lim_1000_seed_23_static_False_steps_50_temp_0.0_view_5`

## Command Line Arguments

Here are the available command line arguments you can use when running this project:

- `--all`: If provided, loads all environments.
- `--API-key`: Specifies the location of your OpenAI API Key. Default: `./utilities/API/API_1`
- `--disp`: If provided and in a GUI environment, displays environment rendering.
- `--envs`: Specifies list of environment names. Refer to `./utilities/envs_mapping.txt` for mapping between index and environment. Default: `["1"]`
- `--exp-msg`: Specifies the path to the hint message for an experience. Default: `./utilities/exp_msg.txt`
- `--exp-src`: Specifies the starting experience read path.
- `--fuc-msg`: Specifies the path to the function helper message. Default: `./utilities/fuc_msg.txt`
- `--goal`: If provided, includes the text goal into the observation description.
- `--gpt`: Specifies the version of GPT to use. Options are "3" or "4". Default: "3"
- `--input`: If provided, action and experience will be given by the user instead of GPT.
- `--lim`: Specifies the word limit for the experience. Default: `300`
- `--log`: If provided, prints logging information.
- `--max-rty`: Specifies the maximum number of delays in OpenAI API Calling. Default: `5`
- `--mry`: Whether the agent will have memory about past experiences.
- `--overwrite`: If provided, overwrites the same experiment setting.
- `--prj-name`: Specifies the project name for your wandb. Default: `LLM As Agent`
- `--rel-des`: If provided, uses relative position description as observation description, otherwise uses pure array print.
- `--rty-dly`: Specifies the number of seconds to delay when in OpenAI API Calling. Default: `1`
- `--rvw-msg`: The review message location to read (default is ./utilities/rvw_msg.txt).
- `--screen`: Sets the resolution for pygame rendering (width and height). Default: `640`
- `--seed`: Sets the random seed for reproducing results. Default: `23`
- `--static`: If provided, the agent will not update the experience during the exploration.
- `--steps`: Sets the maximum number of steps each environment will take. Default: `20`
- `--sys-msg`: Specifies the path to the hint message for an action. Default: `./utilities/sys_msg.txt`
- `--temp`: Specifies the temperature used by the OpenAI API. Default: `0.0`
- `--view`: Sets the number of grid spaces visible in agent-view. Default: `3`
- `--wandb`: If provided, uses wandb to record experiments.
