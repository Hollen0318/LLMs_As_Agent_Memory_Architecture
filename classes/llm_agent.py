import wandb
from utils.log import get_path, write_log, log_desc
from utils.load_data import *
import utils.api.load_api
from utils.exp import initialize_exp, train_exp
from utils.fill import fill_desc_user_0, fill_desc_user_1, fill_reason_user_0
from utils.world_map import update_world_map
from utils.skip import skip
from utils.track import get_track
from utils.alert import examine
from datetime import datetime
from utils.gpt.generate_desc import generate_desc
from utils.gpt.generate_reason import generate_reason
from utils.gpt.generate_action import generate_action

# This is LLM enabled agent class
class agent:

    def __init__(self, args):
        self.args = examine(args)
        # The experience should not be initialized here because experience varies very different during training and evaluation.
        self.init_exp = initialize_exp(args)

    def log(self, texts):
        write_log(self.args, self.save_path, texts)


    def log_desc(self, env_id):
        self.desc_user_0 = fill_desc_user_0(env_id)
        self.desc_user_1 = fill_desc_user_1(env_id, self.pos_x, self.pos_y, self.direction, self.world_map, self.inv, self.past_actions, self.exp, str(lim['desc']))

        self.log(train_msg['desc_sys'])
        self.log(self.desc_user_0)
        self.log(train_msg['desc_assis'])
        self.log(self.desc_user_1)

        return train_msg['desc_sys'], self.desc_user_0, train_msg['desc_assis'], self.desc_user_1

    def get_desc(self, env_id):
        self.desc = generate_desc(self.args, *self.log_desc(env_id), env_id)
        
        self.log(self.desc)

    def log_reason(self):
        self.reason_user_0 = fill_reason_user_0(lim["reason"])
        
        self.log(self.reason_user_0)
        return train_msg['desc_sys'], self.desc_user_0, train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0
    
    def get_reason(self, env_id):
        self.reason = generate_reason(self.args, *self.log_reason(), env_id)
        self.log(self.reason)

    def log_action(self, env_id):
        self.act_user_0 = train_msg["act_user_0"]
        
        self.log(self.act_user_0)
        return train_msg['desc_sys'], self.desc_user_0, train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0, self.reason, self.act_user_0, train_msg["fuc_msg"], train_msg["fuc_desc"]

    def get_action(self, env_id):
        self.action = generate_action(self.args, *self.log_action(), env_id)

        self.log(self.action)

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
            self.exp = train_exp(self.args, env_id, self.init_exp)
            self.inv = 0
            self.past_actions = []
            self.world_map, self.rec = get_track(env_id, env_sizes[str(self.args.seed)])

            if skip(env_id):
                continue

            self.pos_x, self.pos_y, self.direction = self.pos_m[env_id]

            for step in range(self.args.steps[env_id]):
                self.get_desc(env_id)
                self.get_reason(env_id)
                self.get_action(env_id)
                if isinstance(self.action, list):
                    for act in 