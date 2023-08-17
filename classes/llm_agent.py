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
    
    # def add_data(self, img, ):
        #  if self.args.wandb:
        #     self.scn_table.add_data(wandb.Image(img), reason_msg_s, reason, act, n_exp, c_exp)
        #     self.env_table.add_data(str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", ""))
        #     self.obj_table.add_data(str(obj_intr_rec[env_id]), str(obj_view_rec[env_id]))
        #     self.world_map_table.add_data(str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", ""))

        # env_view_ratio, env_step_ratio, env_memo_ratio, obj_intr_ratio, obj_view_ratio = get_ratios(args, env_id, env_view_rec, env_step_rec, env_memo_rec, obj_intr_rec, obj_view_rec)

        # reason_length = count_tokens(reason)
        # with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_reason_{str(reason_length)}.txt"), 'w') as file:
        #     file.write(reason)
        
        # exp_length = count_tokens(exp)
        # with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_exp_{str(exp_length)}.txt"), 'w') as file:
        #     file.write(c_exp)

        # desc_length = count_tokens(desc)
        # with open(os.path.join(save_path, f"env_{i}_idx_{str(j+1)}_desc_{str(desc_length)}.txt"), 'w') as file:
        #     file.write(desc)

        # # Log the data to the dataframe
        # scn_table_df.loc[len(scn_table_df)] = [str(img), reason_msg_s, reason, act, n_exp, c_exp]
        # env_table_df.loc[len(env_table_df)] = [str(env_view_rec[env_id]).replace(".", ""), str(env_step_rec[env_id]).replace(".", ""), str(env_memo_rec[env_id]).replace(".", "")]
        # obj_table_df.loc[len(obj_table_df)] = [str(obj_intr_rec[env_id]), str(obj_view_rec[env_id])]
        # world_map_table_df.loc[len(world_map_table_df)] = [str(world_map[env_id][0]).replace("'", ""), str(world_map[env_id][1]).replace("'", ""), str(world_map[env_id][2]).replace("'", "")]
        # metrics_table_df.loc[len(metrics_table_df)] = [env_view_ratio, env_memo_ratio, env_step_ratio, obj_view_ratio, obj_intr_ratio, exp_length]

        # metrics = {
        #     f"env_{i}/env_view_ratio": env_view_ratio,
        #     f"env_{i}/env_memo_ratio": env_memo_ratio,
        #     f"env_{i}/env_step_ratio": env_step_ratio,
        #     f"env_{i}/obj_view_ratio": obj_view_ratio,
        #     f"env_{i}/obj_intr_ratio": obj_intr_ratio,
        #     f"env_{i}/exp_length": exp_length
        # }

        # if args.wandb:
        #     # Log the metrics
        #     wandb.log(metrics)

    # def add_metric(self, ratio, length):
    #     if self.args.wandb:
    #         # Log the metrics
    #         metrics = {
    #         "env_view_ratio": ratio["env_view"],
    #         "env_memo_ratio": ratio["env_memo"],
    #         "env_step_ratio": ratio["env_step"],
    #         "obj_view_ratio": ratio["obj_view"],
    #         "obj_intr_ratio": ratio["obj_intr"],
    #         "exp_length": length
    #         }
    #         wandb.log(metrics)
        
    #     self.metrics_table_df.loc[len(self.metrics_table_df)] = [ratio["env_view"], ratio["env_memo"], ratio["env_step"], ratio["obj_view"], ratio["obj_intr"], length]

    # def create_table(self):
    #     if self.args.wandb:
    #         self.scn_table = wandb.Table(columns = ["img", "text", "reason", "act", "n_exp", "c_exp"])
    #         self.env_table = wandb.Table(columns = ["env_view_rec", "env_step_rec", "env_memo_rec"])
    #         self.obj_table = wandb.Table(columns = ["obj_intr_rec", "obj_view_rec"])
    #         self.world_map_table = wandb.Table(columns = ["world_map_obj", "world_map_col", "world_map_sta"])
            
    #     # we save them to the csv
    #     # Define table structure
    #     scn_table_columns = ["Image", "Message", "Reason", "Action", "N_exp", "C_exp"]
    #     env_table_columns = ["Env_View", "Env_Step", "Env_Memo"]
    #     obj_table_columns = ["Obj_Intr", "Obj_View"]
    #     world_map_table_columns = ["World_Map_Object", "World_Map_Color", "World_Map_Status"]
    #     metrics_table_columns = ["env_view_ratio", "env_memo_ratio", "env_step_ratio", "obj_view_ratio", "obj_intr_ratio", "exp_length"]

    #     self.scn_table_df = pd.DataFrame(columns=scn_table_columns)
    #     self.env_table_df = pd.DataFrame(columns=env_table_columns)
    #     self.obj_table_df = pd.DataFrame(columns=obj_table_columns)
    #     self.world_map_table_df = pd.DataFrame(columns=world_map_table_columns)
    #     self.metrics_table_df = pd.DataFrame(columns=metrics_table_columns)

    # def save_table(self, env_id):
    #      # Log datas to the wandb
    #     if self.args.wandb:
    #         wandb.log({f"Table/Screenshot for Environment #{env_id}": self.scn_table})
    #         wandb.log({f"Table/Environment Record for Environment #{env_id}": self.env_table})
    #         wandb.log({f"Table/Object Record for Environment #{env_id}": self.obj_table})
    #         wandb.log({f"Table/World Map for Environment #{env_id}": self.world_map_table})

    #     # Save to CSV
    #     self.scn_table_df.to_csv(os.path.join(self.save_path, f'scn_table_env_{env_id}.csv'), index=False)
    #     self.env_table_df.to_csv(os.path.join(self.save_path, f'env_table_env_{env_id}.csv'), index=False)
    #     self.obj_table_df.to_csv(os.path.join(self.save_path, f'obj_table_env_{env_id}.csv'), index=False)
    #     self.world_map_table_df.to_csv(os.path.join(self.save_path, f'world_map_table_env_{env_id}.csv'), index=False)
    #     self.metrics_table_df.to_csv(os.path.join(self.save_path, f'metrics_env_{env_id}.csv'), index=False)

    # def save_image(self, env_id, step):
    #     img_array = self.env.render()
    #     img = Image.fromarray(img_array)
    #     img.save(os.path.join(self.args.save_path, f"env_{env_id}_{step}.png"))

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

    def take_action():
        

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

            for step in range(self.args.steps[env_id]):
                self.update_world_map(env_id)
                self.get_desc(env_id)
                self.get_reason(env_id)
                self.get_action(env_id)
                if isinstance(self.action, list):
                    for act in self.action:
                        act_s = act_obj[str(act)]
