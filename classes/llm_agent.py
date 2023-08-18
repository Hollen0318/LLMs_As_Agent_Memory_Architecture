import wandb
import numpy as np
from utils.log import get_path, write_log
from utils.load_data import *
from utils.exp import initialize_exp, train_exp
from utils.fill import *
from utils.skip import skip
from utils.track import get_track, update_world_map, restore_world_map
from utils.alert import train_examine
from datetime import datetime
from utils.gpt.chat import *
from utils.act.forward import *
from utils.env import start_env, reset_env
from utils.act.left_right import *
from utils.count import *
from PIL import Image
import pandas as pd
import os

# This is LLM enabled agent class
class agent:

    def __init__(self, args):
        if not args.eval:
            self.args = train_examine(args)
        self.init_exp = initialize_exp(args)

    def act_forward(self, env_id):
        self.world_map = restore_world_map(self.world_map, self.pos_x, self.pos_y, self.p_obj, self.p_col, self.p_sta)
        self.pos_x, self.pos_y = update_pos(self.pos_x, self.pos_y, self.direction)
        self.p_obj, self.p_col, self.p_sta, self.world_map = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)

    
    def add_data(self, img, obs, desc, reason, act, n_exp, c_exp, c_world_map):
        if self.args.wandb:
            self.scn_table.add_data(wandb.Image(img), obs, desc, reason, act, n_exp, c_exp)
            self.rec_table.add_data(str(self.rec["env_view"]).replace(".", ""), str(self.rec["env_step"]).replace(".", ""), str(self.rec["env_memo"]).replace(".", ""), str(self.rec["obj_view"]).replace(".", ""), str(self.rec["obj_intr"]["toggle"]).replace(".", ""), str(self.rec["obj_intr"]["pick up"]).replace(".", ""), str(self.rec["obj_intr"]["drop off"]).replace(".", ""))
            self.world_map_table.add_data(str(self.world_map[0]).replace("'", ""), str(self.world_map[1]).replace("'", ""), str(self.world_map[2]).replace("'", ""), str(c_world_map).replace("'", ""))

        self.scn_table_df.loc[len(self.scn_table_df)] = ["Image", obs, desc, reason, act, n_exp, c_exp]
        self.rec_table_df.loc[len(self.rec_table_df)] = [str(self.rec["env_view"]).replace(".", ""), str(self.rec["env_step"]).replace(".", ""), str(self.rec["env_memo"]).replace(".", ""), str(self.rec["obj_view"]).replace(".", ""), str(self.rec["obj_intr"]["toggle"]).replace(".", ""), str(self.rec["obj_intr"]["pick up"]).replace(".", ""), str(self.rec["obj_intr"]["drop off"]).replace(".", "")]
        self.world_map_table_df.loc[len(self.world_map_table_df)] = [str(self.world_map[0]).replace("'", ""), str(self.world_map[1]).replace("'", ""), str(self.world_map[2]).replace("'", ""), str(c_world_map).replace("'", "")]

    def add_length(self, tokens_sum, length, env_id):
        if self.args.wandb:
            # Log the metrics
            length_metrics = {
                f"Environment # {env_id}/tokens_sum": tokens_sum,
                f"Environment # {env_id}/desc_sys": length["desc_sys"],
                f"Environment # {env_id}/desc_user_0": length["desc_user_0"],
                f"Environment # {env_id}/desc_assis": length["desc_assis"],
                f"Environment # {env_id}/desc_user_1": length["desc_user_1"],
                f"Environment # {env_id}/desc": length["desc"],
                f"Environment # {env_id}/reason_user_0": length["reason_user_0"],
                f"Environment # {env_id}/reason": length["reason"],
                f"Environment # {env_id}/n_exp_user_0": length["n_exp_user_0"],
                f"Environment # {env_id}/n_exp": length["n_exp"],
                f"Environment # {env_id}/s_exp_user_0": length["s_exp_user_0"],
                f"Environment # {env_id}/s_exp": length["s_exp"]
            }
            wandb.log(length_metrics)
        
        self.length_table_df.loc[len(self.length_table_df)] = [tokens_sum, length["desc_sys"], length["desc_user_0"], length["desc_assis"], length["desc_user_1"], length["desc"], length["reason_user_0"], length["reason"], length["n_exp_user_0"], length["n_exp"], length["s_exp_user_0"], length["s_exp"]]

    def add_metric(self, ratio, env_id):
        if self.args.wandb:
            # Log the metrics
            metrics = {
            f"Environment # {env_id}/env_view_ratio": ratio["env_view"],
            f"Environment # {env_id}/env_memo_ratio": ratio["env_memo"],
            f"Environment # {env_id}/env_step_ratio": ratio["env_step"],
            f"Environment # {env_id}/obj_view_ratio": ratio["obj_view"],
            f"Environment # {env_id}/toggle_ratio": ratio["toggle"],
            f"Environment # {env_id}/pickup_ratio": ratio["pick up"],
            f"Environment # {env_id}/dropoff_ratio": ratio["drop off"]
            }
            wandb.log(metrics)
        
        self.metrics_table_df.loc[len(self.metrics_table_df)] = [ratio["env_view"], ratio["env_step"], ratio["env_memo"], ratio["obj_view"], ratio["toggle"], ratio["pick up"], ratio["drop off"]]

    def create_table(self):
        if self.args.wandb:
            # This table records the screen shot, the text (observation representation), then the description for that observation representation, reason, action being taken (if it is multiple then we print same act in different screenshots), then we have the new experience (if it is multiple then we print same n_exp), (we will have summarized experience same way too)
            self.scn_table = wandb.Table(columns = ["img", "obs", "desc", "reason", "act", "n_exp", "s_exp"])
            self.rec_table = wandb.Table(columns = ["env_view", "env_step", "env_memo", "obj_view", "toggle", "pick up", "drop off"])
            self.world_map_table = wandb.Table(columns = ["world_map_obj", "world_map_col", "world_map_sta", "c_world_map"])
            
        # we save them to the csv
        # Define table structure
        scn_table_columns = ["img", "obs", "desc", "reason", "act", "n_exp", "s_exp"]
        rec_table_columns = ["env_view", "env_step", "env_memo", "obj_view", "toggle", "pick up", "drop off"]
        world_map_table_columns = ["world_map_obj", "world_map_col", "world_map_sta", "c_world_map"]
        metrics_table_columns = ["env_view", "env_step", "env_memo", "obj_view", "toggle", "pick up", "drop off"]
        length_table_columns = ["sum", "desc_sys", "desc_user_0", "desc_assis", "desc_user_1", "desc", "reason_user_0", "reason", "n_exp_user_0", "n_exp", "s_exp_user_0", "s_sxp"]
        
        self.scn_table_df = pd.DataFrame(columns=scn_table_columns)
        self.rec_table_df = pd.DataFrame(columns=rec_table_columns)
        self.world_map_table_df = pd.DataFrame(columns=world_map_table_columns)
        self.metrics_table_df = pd.DataFrame(columns=metrics_table_columns)
        self.length_table_df = pd.DataFrame(columns=length_table_columns)

    def env_close(self, env_id):
        self.env.close()
        # Log datas to the wandb
        if self.args.wandb:
            wandb.log({f"Table/Screenshot Table for Environment #{env_id}": self.scn_table})
            wandb.log({f"Table/Record Table for Environment #{env_id}": self.rec_table})
            wandb.log({f"Table/World Map for Environment #{env_id}": self.world_map_table})

        # Save to CSV
        self.scn_table_df.to_csv(os.path.join(self.save_path, f'scn_table_env_{env_id}.csv'), index = True)
        self.rec_table_df.to_csv(os.path.join(self.save_path, f'rec_table_env_{env_id}.csv'), index = True)
        self.world_map_table_df.to_csv(os.path.join(self.save_path, f'world_map_table_env_{env_id}.csv'), index = True)
        self.metrics_table_df.to_csv(os.path.join(self.save_path, f'metrics_env_{env_id}.csv'), index = True)
        self.length_table_df.to_csv(os.path.join(self.save_path, f"length_table_env_{env_id}.csv"), index = True)

    def execute_action(self, act, env_id):
        if act == "left":
            # We update the world map, env view, env memo, obj_view, act history, arrow
            self.direction = left_arrow(self.direction)
            self.obs, _, terminated, _, _ = self.env.step(Actions.left)
            self.past_actions.append(act)
            _, _, _, self.world_map = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)

        elif act == "right":
            self.direction = right_arrow(self.direction)
            self.obs, _, terminated, _, _ = self.env.step(Actions.right)
            self.past_actions.append(act)
            _, _, _, self.world_map = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)

        elif act == "forward":
            self.obs, _, self.terminated, _, _ = self.env.step(Actions.forward)
            self.past_actions.append(act)
            if self.terminated:
                self.inv = 0
                self.world_map = restore_world_map(self.world_map, self.pos_x, self.pos_y, self.p_obj, self.p_col, self.p_sta)
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                self.pos_x, self.pos_y, self.direction = self.pos_m[str(env_id)]
                # Initilize the environment
                self.obs, _ = reset_env(self.env, self.args.seed)
                self.p_obj, self.p_col, self.p_sta = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)

                front_obj = get_front_obj(self.world_map, self.pos_x, self.pos_y,self. arrow)
                # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
                if front_obj == 8:
                    self.n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                elif front_obj == 9:
                    self.n_exp = "You touched a lava and dead, respawn at the start place, try again!"
                elif front_obj == 6:
                    self.n_exp = "You are killed stepping towards this ball"
                self.log(self.n_exp)
            else:
                front_obj = get_front_obj(self.world_map, self.pos_x, self.pos_y, self.direction)
                front_sta = get_front_sta(self.world_map, self.pos_x, self.pos_y, self.direction)
                if front_obj == 1 or front_obj == 3:
                    self.act_forward(env_id)
                elif front_obj == 4 and front_sta == 0:
                    self.act_forward(env_id)

        elif act == "toggle": 
            self.obs, _, self.terminated, _, _ = self.env.step(Actions.toggle)
            self.past_actions.append(act)
            if self.terminated:
                self.inv = 0
                self.world_map = restore_world_map(self.world_map, self.pos_x, self.pos_y, self.p_obj, self.p_col, self.p_sta)
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                self.pos_x, self.pos_y, self.direction = self.pos_m[str(env_id)]
                # Initilize the environment
                self.obs, _ = reset_env(self.env, self.args.seed)
                self.p_obj, self.p_col, self.p_sta = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)
                self.n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                self.log(self.n_exp)
            else:
                _, _, _, self.world_map = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)
                front_obj = get_front_obj(self.world_map, self.pos_x, self.pos_y, self.direction)
                self.rec["obj_intr"]["toggle"][front_obj] += 1

        elif act == "drop off":
            self.n_obs, _, self.terminated, _, _ = self.env.step(Actions.drop)
            self.past_actions.append(act)
            if self.terminated:
                self.inv = 0
                self.world_map = restore_world_map(self.world_map, self.pos_x, self.pos_y, self.p_obj, self.p_col, self.p_sta)
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                self.pos_x, self.pos_y, self.direction = self.pos_m[str(env_id)]
                # Initilize the environment
                self.obs, _ = reset_env(self.env, self.args.seed)
                self.p_obj, self.p_col, self.p_sta = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)
                self.n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                self.log(self.n_exp)
            else:
                _, _, _, self.world_map = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)
                if not np.array_equal(self.n_obs['image'].transpose(1,0,2), self.obs['image'].transpose(1,0,2)):
                    self.inv = 0
                front_obj = get_front_obj(self.world_map, self.pos_x, self.pos_y, self.direction)
                self.obs = self.n_obs.copy()
                self.rec["obj_intr"]["drop off"][front_obj] += 1
            
        elif act == "pick up":
            self.n_obs, _, self.terminated, _, _ = self.env.step(Actions.drop)
            self.past_actions.append(act)
            if self.terminated:
                self.inv = 0
                self.world_map = restore_world_map(self.world_map, self.pos_x, self.pos_y, self.p_obj, self.p_col, self.p_sta)
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                self.pos_x, self.pos_y, self.direction = self.pos_m[str(env_id)]
                # Initilize the environment
                self.obs, _ = reset_env(self.env, self.args.seed)
                self.p_obj, self.p_col, self.p_sta = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)
                self.n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                self.log(self.n_exp)
            else:
                _, _, _, self.world_map = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)
                if not np.array_equal(self.n_obs['image'].transpose(1,0,2), self.obs['image'].transpose(1,0,2)):
                    self.inv = front_obj
                front_obj = get_front_obj(self.world_map, self.pos_x, self.pos_y, self.direction)
                self.obs = self.n_obs.copy()
                self.rec["obj_intr"]["pick up"][front_obj] += 1

        # We refresh the action history every args.refresh run to avoid too large action space
        if len(self.past_actions) >= lim["memo"]:
            self.past_actions = self.past_actions[1:]


    def get_action(self, env_id):
        self.action = generate_action(self.args, *self.log_action(), env_id)

    def get_desc(self, env_id):
        self.desc = generate_desc(self.args, *self.log_desc(env_id), env_id)
        
        self.log(self.desc)
    
    def get_length(self):
        length = {}
        length["desc_sys"] = tokens_count(train_msg["desc_sys"])
        length["desc_user_0"] = tokens_count(train_msg["desc_user_0"])
        length["desc_assis"] = tokens_count(train_msg["desc_assis"])
        length["desc_user_1"] = tokens_count(self.desc_user_1)
        length["desc"] = tokens_count(self.desc)
        length["reason_user_0"] = tokens_count(self.reason_user_0)
        length["reason"] = tokens_count(self.reason)
        length["n_exp_user_0"] = tokens_count(self.n_exp_user_0)
        length["n_exp"] = tokens_count(self.n_exp)
        length["s_exp_user_0"] = tokens_count(self.s_exp_user_0)
        length["s_exp"] = tokens_count(self.exp)
        tokens_sum = sum(length.values())
        return tokens_sum, length

    def get_metrics(self):
        env_view_r = np.count_nonzero(self.rec["env_view"]) / np.size(self.rec["env_view"]) * 100
        env_view_r_s = "{:.3f}%".format(env_view_r)

        env_step_r = np.count_nonzero(self.rec["env_step"]) / np.size(self.rec["env_step"]) * 100
        env_step_r_s = "{:.3f}%".format(env_step_r)

        env_memo_r = np.count_nonzero(self.rec["env_memo"]) / np.size(self.rec["env_memo"]) * 100
        env_memo_r_s = "{:.3f}%".format(env_memo_r)

        toggle_r = np.count_nonzero(self.rec["obj_intr"]["toggle"]) / np.size(self.rec["obj_intr"]["toggle"]) * 100
        toggle_r_s = "{:.3f}%".format(toggle_r)
        
        pickup_r = np.count_nonzero(self.rec["obj_intr"]["pick up"]) / np.size(self.rec["obj_intr"]["pick up"]) * 100
        pickup_r_s = "{:.3f}%".format(pickup_r)
        
        dropoff_r = np.count_nonzero(self.rec["obj_intr"]["drop off"]) / np.size(self.rec["obj_intr"]["drop off"]) * 100
        dropoff_r_s = "{:.3f}%".format(dropoff_r)

        obj_view_r = np.count_nonzero(self.rec["obj_view"]) / np.size(self.rec["obj_view"]) * 100
        obj_view_r_s = "{:.3f}%".format(obj_view_r)

        ratio = {}

        ratio["env_view"] = env_view_r
        ratio["env_step"] = env_step_r
        ratio["env_memo"] = env_memo_r
        ratio["obj_view"] = obj_view_r
        ratio["toggle"] = toggle_r
        ratio["pick up"] = pickup_r
        ratio["drop off"] = dropoff_r

        ratio_s = {}

        ratio_s["env_view"] = env_view_r_s
        ratio_s["env_step"] = env_step_r_s
        ratio_s["env_memo"] = env_memo_r_s
        ratio_s["obj_view"] = obj_view_r_s
        ratio_s["toggle"] = toggle_r_s
        ratio_s["pick up"] = pickup_r_s
        ratio_s["drop off"] = dropoff_r_s

        self.log_metrics(ratio_s)

        return ratio

    def get_n_exp(self, env_id):
        self.n_exp = generate_n_exp(self.args, *self.log_n_exp(env_id), env_id)

        self.log(self.n_exp)

    def get_reason(self, env_id):
        self.reason = generate_reason(self.args, *self.log_reason(), env_id)
        self.log(self.reason)

    def get_s_exp(self, env_id):
        self.exp = generate_s_exp(self.args, *self.log_s_exp(), env_id)
        self.log(self.exp)

    def log(self, texts):
        write_log(self.args, self.save_path, texts)

    def log_action(self):
        self.act_user_0 = train_msg["act_user_0"]
        
        self.log(self.act_user_0)
        return train_msg['desc_sys'], train_msg["desc_user_0"], train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0, self.reason, self.act_user_0, train_msg["fuc_msg"], train_msg["fuc_desc"]
    
    def log_desc(self, env_id):
        self.desc_user_1 = fill_desc_user_1(env_id, self.pos_x, self.pos_y, self.direction, self.world_map, self.inv, self.past_actions, self.exp, lim['desc'])

        self.log(train_msg['desc_sys'])
        self.log(train_msg["desc_user_0"])
        self.log(train_msg['desc_assis'])
        self.log(self.desc_user_1)

        return train_msg['desc_sys'], train_msg["desc_user_0"], train_msg['desc_assis'], self.desc_user_1
    
    def log_metrics(self, ratio_s):
        rpt_msg_f = rpt_msg["rpt_msg"]
        rpt_msg_s = rpt_msg_f.format(ratio_s["env_view"], ratio_s["env_step"], ratio_s["env_memo"], ratio_s["obj_view"], ratio_s["toggle"], ratio_s["pick up"], ratio_s["drop off"])
        self.log(rpt_msg_s)

    def log_n_exp(self, env_id):
        self.n_exp_user_0 = fill_n_exp_user_0(self.act_l, env_id, self.pos_x, self.pos_y, self.direction, self.world_map, self.inv, self.past_actions, lim["n_exp"])

        self.log(self.n_exp_user_0)
        return train_msg['desc_sys'], train_msg["desc_user_0"], train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0, self.reason, self.n_exp_user_0

    def log_reason(self):
        self.reason_user_0 = fill_reason_user_0(lim["reason"])
        
        self.log(self.reason_user_0)
        return train_msg['desc_sys'], train_msg["desc_user_0"], train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0

    def log_s_exp(self):
        self.s_exp_user_0 = fill_s_exp_user_0(lim["s_exp"])

        self.log(self.s_exp_user_0)
        return train_msg['desc_sys'], train_msg["desc_user_0"], train_msg['desc_assis'], self.desc_user_1, self.desc, self.reason_user_0, self.reason, self.n_exp_user_0, self.n_exp, self.s_exp_user_0

    def save_table(self, env_id):
         # Log datas to the wandb
        if self.args.wandb:
            wandb.log({f"Table/Screenshot for Environment #{env_id}": self.scn_table})
            wandb.log({f"Table/Environment Record for Environment #{env_id}": self.rec_table})
            wandb.log({f"Table/World Map for Environment #{env_id}": self.world_map_table})

        # Save to CSV
        self.scn_table_df.to_csv(os.path.join(self.save_path, f'scn_table_env_{env_id}.csv'), index=True)
        self.rec_table_df.to_csv(os.path.join(self.save_path, f'env_table_env_{env_id}.csv'), index=True)
        self.world_map_table_df.to_csv(os.path.join(self.save_path, f'world_map_table_env_{env_id}.csv'), index=True)
        self.metrics_table_df.to_csv(os.path.join(self.save_path, f'metrics_table_env_{env_id}.csv'), index=True)
        self.length_table_df.to_csv(os.path.join(self.save_path, f'length_table_env_{env_id}.csv'), index=True)


    def save_image(self, env_id, step):
        img_array = self.env.render()
        img = Image.fromarray(img_array)
        img.save(os.path.join(self.save_path, f"env_{env_id}_{step}.png"))
        return img

    def save_wandb(self):
        if self.args.wandb:
            wandb.finish()

    def train(self):
        if self.args.wandb:
            wandb.init(
                project = self.args.prj_name,
                name = datetime.now().strftime(r"Train %Y-%m-%d %H:%M:%S"),
                config = vars(self.args)
            )
        
        if self.args.all:
            self.envs = [i for i in range(len(env_ids))]

        # The self.pos_m stores the env position pos_x, pos_y, direction in the args.seed for all envs
        self.pos_m = get_pos_m(env_pos[str(self.args.seed)])

        for env_id in range(len(self.args.envs)):
            self.save_path = get_path(self.args, env_id)
            # We need to determine the save path before print the configurations
            self.log(f"################## Starting Experiment ##################\n")
            self.log(f"Configurations are:\n{self.args}\n")
            # This self.exp is the experience particular for this env
            self.exp = train_exp(self.args, env_id, self.init_exp)
            self.inv = 0
            self.past_actions = []
            # The self.world_map, self.rec stores the world map for this particular env
            self.world_map, self.rec = get_track(self.args.envs[env_id], env_sizes[str(self.args.seed)])
            if skip(env_id):
                continue
            
            # We obtain the starting position for this environment by using index [env_id]
            self.pos_x, self.pos_y, self.direction = self.pos_m[str(self.args.envs[env_id])]

            # Start the environment
            self.env = start_env(self.args, env_id)
            self.obs, _ = reset_env(self.env, self.args.seed)
            self.terminated = False
            self.p_obj, self.p_col, self.p_sta, self.world_map = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec, env_id)

            self.create_table()

            for step in range(self.args.steps[env_id]):
                self.get_desc(env_id)
                self.get_reason(env_id)
                self.get_action(env_id)
                if isinstance(self.action, list):
                    img_l = []
                    metric_l = []
                    self.act_l = []
                    for act_int in self.action:
                        img_l.append(self.save_image(self.args.envs[env_id], step))
                        metric_l.append(self.get_metrics())
                        act = int_act[str(act_int)]
                        self.act_l.append(act)
                        # World map updated inside the execute_action
                        self.execute_action(act, self.args.envs[env_id])
                        if self.terminated:
                            step += 1
                            break
                        step += 1
                        # TODO write the exp, desc into some files, count the length of experience, desc, reason, etc. 
                        # TODO wandb table
                        # The action is integer, not string at the moment
                    if not self.terminated:
                        self.get_n_exp(env_id)
                        self.get_s_exp(env_id)
                    else:
                        self.terminated = False
                        self.get_s_exp(env_id)
                        
                    for i in range(len(self.action)):
                        self.add_data(img_l[i], self.desc_user_1, self.desc, self.reason, str(self.act_l), self.n_exp, self.exp, compose_world_map(self.world_map))
                        self.add_metric(metric_l[i], env_id)
                    self.add_length(*self.get_length(), self.args.envs[env_id])
                elif isinstance(self.action, int):
                    self.act_l = []
                    act = int_act[str(self.action)]
                    self.act_l.append(act)
                    img = self.save_image(self.args.envs[env_id], step)
                    metrics = self.get_metrics()
                    self.execute_action(act, self.args.envs[env_id])
                    if not self.terminated:
                        self.get_n_exp(env_id)
                        self.get_s_exp(env_id)
                    else:
                        self.terminated = False
                        self.get_s_exp(env_id)
                    self.add_data(img, self.desc_user_1, self.desc, self.reason, act, self.n_exp, self.exp, compose_world_map(self.world_map))
                    self.add_metric(metrics, env_id)
                    self.add_length(*self.get_length(), self.args.envs[env_id])
            self.env_close(env_id)
            self.save_table(env_id)

        self.save_wandb()