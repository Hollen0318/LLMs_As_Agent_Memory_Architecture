import wandb
from utils.log import get_path, write_log, log_desc
from utils.load_data import *
from utils.exp import initialize_exp, train_exp
from utils.fill import fill_desc_user_0, fill_desc_user_1, fill_reason_user_0
from utils.skip import skip
from utils.track import get_track, update_world_map
from utils.alert import examine
from datetime import datetime
from utils.gpt.generate_desc import generate_desc
from utils.gpt.generate_reason import generate_reason
from utils.gpt.generate_action import generate_action
from utils.env import start_env, reset_env
from PIL import Image
import pandas as pd
import os

# This is LLM enabled agent class
class agent:

    def __init__(self, args):
        self.args = examine(args)
        # The experience should not be initialized here because experience varies very different during training and evaluation.
        self.init_exp = initialize_exp(args)
    
    def add_data(self, img, obs, desc, reason, act, n_exp, c_exp, c_world_map):
        if self.args.wandb:
            self.scn_table.add_data(wandb.Image(img), obs, desc, reason, act, n_exp, c_exp)
            self.rec_table.add_data(str(self.rec["env_view"]).replace(".", ""), str(self.rec["env_step"]).replace(".", ""), str(self.rec["env_memo"]).replace(".", ""), str(self.rec["obj_view"]).replace(".", ""), str(self.rec["obj_intr"]).replace(".", ""))
            self.world_map_table.add_data(str(self.world_map[0]).replace("'", ""), str(self.world_map[1]).replace("'", ""), str(self.world_map[2]).replace("'", ""), str(c_world_map).replace("'", ""))

        self.scn_table_df.loc[len(self.scn_table_df)] = ["Image", obs, desc, reason, act, n_exp, c_exp]
        self.rec_table_df.loc[len(self.rec_table_df)] = [str(self.rec["env_view"]).replace(".", ""), str(self.rec["env_step"]).replace(".", ""), str(self.rec["env_memo"]).replace(".", ""), str(self.rec["obj_view"]).replace(".", ""), str(self.rec["obj_intr"]).replace(".", "")]
        self.world_map_table_df.loc[len(self.world_map_table_df)] = [str(self.world_map[0]).replace("'", ""), str(self.world_map[1]).replace("'", ""), str(self.world_map[2]).replace("'", ""), str(c_world_map).replace("'", "")]

    def add_metric(self, ratio, length):
        if self.args.wandb:
            # Log the metrics
            metrics = {
            "env_view_ratio": ratio["env_view"],
            "env_memo_ratio": ratio["env_memo"],
            "env_step_ratio": ratio["env_step"],
            "obj_view_ratio": ratio["obj_view"],
            "obj_intr_ratio": ratio["obj_intr"],
            "exp_length": length
            }
            wandb.log(metrics)
        
        self.metrics_table_df.loc[len(self.metrics_table_df)] = [ratio["env_view"], ratio["env_memo"], ratio["env_step"], ratio["obj_view"], ratio["obj_intr"], length]

    def create_table(self):
        if self.args.wandb:
            # This table records the screen shot, the text (observation representation), then the description for that observation representation, reason, action being taken (if it is multiple then we print same act in different screenshots), then we have the new experience (if it is multiple then we print same n_exp), (we will have summarized experience same way too)
            self.scn_table = wandb.Table(columns = ["img", "obs", "desc", "reason", "act", "n_exp", "s_exp"])
            self.rec_table = wandb.Table(columns = ["env_view", "env_step", "env_memo", "obj_view", "obj_intr"])
            self.world_map_table = wandb.Table(columns = ["world_map_obj", "world_map_col", "world_map_sta", "c_world_map"])
            
        # we save them to the csv
        # Define table structure
        scn_table_columns = ["img", "obs", "desc", "reason", "act", "n_exp", "s_exp"]
        rec_table_columns = ["env_view", "env_step", "env_memo", "obj_view", "obj_intr"]
        world_map_table_columns = ["world_map_obj", "world_map_col", "world_map_sta", "c_world_map"]
        metrics_table_columns = ["env_view", "env_memo", "env_step", "obj_view", "obj_intr", "exp_length"]

        self.scn_table_df = pd.DataFrame(columns=scn_table_columns)
        self.env_table_df = pd.DataFrame(columns=rec_table_columns)
        self.world_map_table_df = pd.DataFrame(columns=world_map_table_columns)
        self.metrics_table_df = pd.DataFrame(columns=metrics_table_columns)

    def execute_action(self, ):
        

    def get_action(self, env_id):
        self.action = generate_action(self.args, *self.log_action(), env_id)

        self.log(self.action)

    def get_desc(self, env_id):
        self.desc = generate_desc(self.args, *self.log_desc(env_id), env_id)
        
        self.log(self.desc)
    
    def get_reason(self, env_id):
        self.reason = generate_reason(self.args, *self.log_reason(), env_id)
        self.log(self.reason)

    def log(self, texts):
        write_log(self.args, self.save_path, texts)

    def log_action(self):
        self.act_user_0 = train_msg["act_user_0"]
        
        self.log(self.act_user_0)
        return train_msg['desc_sys'], self.desc_user_0, train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0, self.reason, self.act_user_0, train_msg["fuc_msg"], train_msg["fuc_desc"]
    
    def log_desc(self, env_id):
        self.desc_user_0 = fill_desc_user_0(env_id)
        self.desc_user_1 = fill_desc_user_1(env_id, self.pos_x, self.pos_y, self.direction, self.world_map, self.inv, self.past_actions, self.exp, str(lim['desc']))

        self.log(train_msg['desc_sys'])
        self.log(self.desc_user_0)
        self.log(train_msg['desc_assis'])
        self.log(self.desc_user_1)

        return train_msg['desc_sys'], self.desc_user_0, train_msg['desc_assis'], self.desc_user_1

    def log_reason(self):
        self.reason_user_0 = fill_reason_user_0(lim["reason"])
        
        self.log(self.reason_user_0)
        return train_msg['desc_sys'], self.desc_user_0, train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0

    def save_table(self, env_id):
         # Log datas to the wandb
        if self.args.wandb:
            wandb.log({f"Table/Screenshot for Environment #{env_id}": self.scn_table})
            wandb.log({f"Table/Environment Record for Environment #{env_id}": self.env_table})
            wandb.log({f"Table/Object Record for Environment #{env_id}": self.obj_table})
            wandb.log({f"Table/World Map for Environment #{env_id}": self.world_map_table})

        # Save to CSV
        self.scn_table_df.to_csv(os.path.join(self.save_path, f'scn_table_env_{env_id}.csv'), index=False)
        self.env_table_df.to_csv(os.path.join(self.save_path, f'env_table_env_{env_id}.csv'), index=False)
        self.obj_table_df.to_csv(os.path.join(self.save_path, f'obj_table_env_{env_id}.csv'), index=False)
        self.world_map_table_df.to_csv(os.path.join(self.save_path, f'world_map_table_env_{env_id}.csv'), index=False)
        self.metrics_table_df.to_csv(os.path.join(self.save_path, f'metrics_env_{env_id}.csv'), index=False)

    def save_image(self, env_id, step):
        img_array = self.env.render()
        img = Image.fromarray(img_array)
        img.save(os.path.join(self.args.save_path, f"env_{env_id}_{step}.png"))
        return img

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

            # Start the environment
            self.env = start_env(self.args, env_id)
            self.obs, _ = reset_env(self.env, self.args.seed)

            self.p_obj, self.p_col, self.p_sta = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec)

            self.create_table()

            for step in range(self.args.steps[env_id]):
                self.update_world_map(env_id)
                self.get_desc(env_id)
                self.get_reason(env_id)
                self.get_action(env_id)
                
                if isinstance(self.action, list):
                    for act in self.action:
                        act_s = act_obj[str(act)]
                        self.execute_action(act_s)

                img = self.save_image(env_id, 0)
                self.add_data(img, self.desc_user_1, self.desc, self.reason, str(self.act), )