import wandb
from utils.log import get_path, write_log
from utils.load_data import *
import utils.api.load_api
from utils.skip import skip
from utils.track import get_track
from datetime import datetime

# This is LLM enabled agent class
class agent:

    def __init__(self, args):
        self.args = args
        self.gpt = gpt_map[args.gpt]
        # Initiate the starting experience
        if args.exp_src is not None:
            self.exp = open(args.exp_src).read()
        else:
            self.exp = ""

    def log(self, texts):
        write_log(self.args, self.save_path, texts)

    def train(self):

        if self.args.wandb:
            wandb.init(
                project = self.args.prj_name,
                name = datetime.now().strftime(r"Train %Y-%m-%d %H:%M:%S"),
                config = vars(self.args)
            )
        
        # Getting the environment list
        if self.args.all:
            self.envs = [str(i) for i in range()]

        # For each seed, we will have new world maps and record because different seed has different layout
        # For each environment, we obtain its unique world map and rec
        for seed in self.args.seeds:
            self.pos_m = get_pos_m(env_pos[str(seed)])
            for env_id in self.args.envs:
                self.save_path = get_path(self.args, seed, env_id)
                # We need to determine the save path before print the configurations
                self.log(f"################## Starting Experiment ##################\n")
                self.log(f"Configurations are:\n{self.args}\n")

                self.world_map, self.rec = get_track(env_id, env_sizes[str(seed)])
                if not skip(env_id):
                    