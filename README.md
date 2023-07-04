# LLM_As_Agent
Project Repository for LLM As Agent project code

## Getting Started
1. Test enviornment 0, print the logging infos, stored in wandb, with system message being ./utilities/n_sys_msg.tx, overwritting the same settings, experience limit being 1000 tokens.
```shell
python start.py --log --sys-msg ./utilities/n_sys_msg.txt --envs 0 --overwrite --lim 1000
```