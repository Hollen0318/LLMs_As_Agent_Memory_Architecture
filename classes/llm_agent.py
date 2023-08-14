import wandb
from utils.log import get_path, write_log
from utils.load_data import *
import utils.api.load_api
from utils.fill import *
from utils.world_map import update_world_map
from utils.skip import skip
from utils.track import get_track
from utils.alert import examine
from datetime import datetime

# This is LLM enabled agent class
class agent:

    def __init__(self, args):
        self.args = examine(args)
        # Initiate the starting experience

    def log(self, texts):
        write_log(self.args, self.save_path, texts)

    def desc(self, env_id):
        self.desc_user_0 = fill_desc_user_0(env_id)
        self.log(f"\n################## System Message ##################\n")
        self.log(train_msg['desc_sys'])
        self.log(self.sys_msg)
        self.log(train_msg['desc_assis'])
        self.desc_user_1 = fill_desc_user_1(self.args, env_id, self.pos_x, self.pos_y, self.direction, self.world_map, self.inv, self.past_actions, lim['desc'])

    def train(self):
        if self.args.wandb:
            wandb.init(
                project = self.args.prj_name,
                name = datetime.now().strftime(r"Train %Y-%m-%d %H:%M:%S"),
                config = vars(self.args)
            )
        
        # Getting the environment list
        if self.args.all:
            self.envs = [i for i in range(len(env_ids))]

        # For each seed, we will have new world maps and record because different seed has different layout
        # For each environment, we obtain its unique world map and rec
        self.pos_m = get_pos_m(env_pos[str(self.args.seed)])

        for env_id in self.args.envs:
            self.save_path = get_path(self.args, env_id)
            # We need to determine the save path before print the configurations
            self.log(f"################## Starting Experiment ##################\n")
            self.log(f"Configurations are:\n{self.args}\n")

            self.inv = 0
            self.past_actions = []
            self.world_map, self.rec = get_track(env_id, env_sizes[str(self.args.seed)])

            if skip(env_id):
                continue

            self.pos_x, self_pos_y = self.pos_m[]

            for step in range(self.args.steps[env_id]):
                self.desc_response = self.desc(env_id)
