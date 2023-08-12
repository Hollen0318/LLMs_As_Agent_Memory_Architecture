from utils.log import *
from utils.load_data import *
from utils.api import load_api

# This is LLM enabled agent class
class agent:

    def __init__(self, args):
        self.args = args
        self.save_path = get_path(args)
        self.gpt = gpt_map[args.gpt]
        
        # Initiate the starting experience
        if args.exp_src is not None:
            self.exp = open(args.exp_src).read()
        else:
            self.exp = ""
        
    def log(self, texts):
        write_log(self.args, self.save_path, texts)

    def train(self):
        self.write_log(f"################## Starting Experiment ##################\n")
        self.write_log(f"Configurations are:\n{self.args}\n")

        if self.args.wandb:
            wandb.init(
                project = self.args.prj_name,
                name = datetime.now().strftime(r"Train %Y-%m-%d %H:%M:%S"),
                config = vars(self.args)
            )
        
        # Getting the environment list
        if self.args.all:
            self.envs = [str(i) for i in range()]