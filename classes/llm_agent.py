import wandb
from utils.log import get_path, write_log
from utils.load_data import gpt_map, env_ids, data, goals, train_rec
import utils.api.load_api
from utils.track import get_world_maps
from datetime import datetime

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

        # For each seed, we will have new world map
        self.world_map = get_world_maps(env_id, data["env_sizes_" + str(self.args.seed)])
        self.rec = get_rec                                                                