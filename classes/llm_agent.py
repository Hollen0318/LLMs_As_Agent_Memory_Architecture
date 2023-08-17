import wandb
from utils.log import get_path, write_log
from utils.load_data import *
from utils.exp import initialize_exp, train_exp
from utils.fill import *
from utils.skip import skip
from utils.track import get_track, update_world_map
from utils.alert import train_examine
from datetime import datetime
from utils.gpt.chat import *
from utils.env import start_env, reset_env
from utils.act.act_control import *
from PIL import Image
import pandas as pd
import os

# This is LLM enabled agent class
class agent:

    def __init__(self, args):
        if not args.eval:
            self.args = train_examine(args)
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

    def execute_action(self, act, env_id):
        if act == "left":
            # We update the world map, env view, env memo, obj_view, act history, arrow
            self.direction = left_arrow(self.direction)
            self.obs, _, terminated, _, _ = self.env.step(Actions.left)
            self.past_actions.append(act)
            _, _, _ = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec)

        elif act == "right":
            self.direction = right_arrow(self.direction)
            self.obs, _, terminated, _, _ = self.env.step(Actions.right)
            self.past_actions.append(act)
            _, _, _ = update_world_map(self.args, self.world_map, self.pos_x, self.pos_y, self.direction, self.obs, self.rec)

        elif act == "forward":
            self.obs, _, self.terminated, _, _ = self.env.step(Actions.forward)
            self.past_actions.append(act)
            if self.terminated:
                inv = 0
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                self.pos_x, self.pos_y, self.direction = self.pos_m[env_id]
                # Initilize the environment
                self.obs, _ = reset_env(self.env, self.args.seed)
                front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
                write_log(args, save_path, f"\n*************************************************\n\nDoing forward, the p_obj and p_col, p_sta is {p_obj} {p_col} {p_sta } *************************************\n")
                world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
                # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
                p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                if front_obj == 8:
                    n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                elif front_obj == 9:
                    n_exp = "You touched a lava and dead, respawn at the start place, try again!"
                elif front_obj == 6:
                    n_exp = "You are killed stepping towards this ball"
                c_exp = sum_exp(args, n_exp, exp, act_his)
                exp = c_exp
                pos_x = n_pos_x
                pos_y = n_pos_y
                arrow = n_arrow
                return 
            else:
                front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
                front_col = get_front_col(args, env_id, world_map, pos_x, pos_y, arrow)
                front_sta = get_front_sta(args, env_id, world_map, pos_x, pos_y, arrow)

                if front_obj == 1 or front_obj == 3:
                    n_pos_x, n_pos_y = update_pos(pos_x, pos_y, arrow)
                elif front_obj == 4 and front_sta == 0:
                    n_pos_x, n_pos_y = update_pos(pos_x, pos_y, arrow)
                else:
                    n_pos_x, n_pos_y = pos_x, pos_y
            write_log(args, save_path, f"\n*************************************************\n\nDoing forward, the p_obj and p_col, p_sta is {p_obj} {p_col} {p_sta } *************************************\n")
            world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
            p_obj, p_col, p_sta = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his, pos_x, pos_y, n_pos_x, n_pos_y, arrow, arrow)
            c_exp = sum_exp(args, n_exp, exp, act_his)

            exp = c_exp
            obs = n_obs

            pos_x = n_pos_x
            pos_y = n_pos_y

        elif act == "toggle": 
            o_world_map = {}
            o_world_map[env_id] = world_map[env_id].copy()
            o_world_map[env_id][0] = world_map[env_id][0].copy()
            o_world_map[env_id][1] = world_map[env_id][1].copy()
            o_world_map[env_id][2] = world_map[env_id][2].copy()
            front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
            if front_obj != 7:
                n_obs, reward, terminated, truncated, _ = env.step(Actions.toggle)
            else:
                n_obs, reward, terminated, truncated, _ = obs, 0.0, False, False, _
            act_his.append(act)
            if terminated:
                # For each new environment, the inventory is always 0
                inv = 0
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                n_pos_x, n_pos_y, n_arrow = pos_m[env_id]
                # Initilize the environment
                obs, state = env.reset(seed=args.seed)
                world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
                # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
                _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                c_exp = sum_exp(args, n_exp, exp, act_his)
                exp = c_exp
                pos_x = n_pos_x
                pos_y = n_pos_y
                arrow = n_arrow
                return
            write_log(args, save_path, f"The front object being interacted with is {front_obj}")
            obj_intr_rec[env_id][0][front_obj] += 1
            _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, arrow)
            c_exp = sum_exp(args, n_exp, exp, act_his)

            exp = c_exp
            obs = n_obs

        elif act == "drop off":
            o_world_map = {}
            o_world_map[env_id] = world_map[env_id].copy()
            o_world_map[env_id][0] = world_map[env_id][0].copy()
            o_world_map[env_id][1] = world_map[env_id][1].copy()
            o_world_map[env_id][2] = world_map[env_id][2].copy()
            n_obs, reward, terminated, truncated, _ = env.step(Actions.drop)
            act_his.append(act)
            if terminated:
                # For each new environment, the inventory is always 0
                inv = 0
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                n_pos_x, n_pos_y, n_arrow = pos_m[env_id]
                # Initilize the environment
                obs, state = env.reset(seed=args.seed)
                front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
                world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
                # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
                _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                c_exp = sum_exp(args, n_exp, exp, act_his)
                exp = c_exp
                pos_x = n_pos_x
                pos_y = n_pos_y
                arrow = n_arrow
                return 
            else:
                if not np.array_equal(n_obs['image'].transpose(1,0,2), obs['image'].transpose(1,0,2)):
                    n_inv = 0
                else:
                    n_inv = inv
            front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
            write_log(args, save_path, f"The front object being interacted with is {front_obj}")
            obj_intr_rec[env_id][1][front_obj] += 1
            _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, n_inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, arrow)
            c_exp = sum_exp(args, n_exp, exp, act_his)

            exp = c_exp
            obs = n_obs
            inv = n_inv
            
        elif act == "pick up":
            o_world_map = {}
            o_world_map[env_id] = world_map[env_id].copy()
            o_world_map[env_id][0] = world_map[env_id][0].copy()
            o_world_map[env_id][1] = world_map[env_id][1].copy()
            o_world_map[env_id][2] = world_map[env_id][2].copy()
            n_obs, reward, terminated, truncated, _ = env.step(Actions.pickup)
            act_his.append(act)
            if terminated:
                # For each new environment, the inventory is always 0
                inv = 0
                # Get the respawn position for seed = 23 only pos_x and pos_y are integer indicating the coordinates, arrow is string like a Right
                n_pos_x, n_pos_y, n_arrow = pos_m[env_id]
                # Initilize the environment
                obs, state = env.reset(seed=args.seed)
                front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
                world_map[env_id][0][pos_x][pos_y], world_map[env_id][1][pos_x][pos_y], world_map[env_id][2][pos_x][pos_y] = p_obj, p_col, p_sta
                # We update the world map, environment view, step, memo and object view to be consistent with the environment obs.
                _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, n_pos_x, n_pos_y, n_arrow, obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
                n_exp = "You completed a hidden goal and is sent back to start place, congratulations!"
                c_exp = sum_exp(args, n_exp, exp, act_his)
                exp = c_exp
                pos_x = n_pos_x
                pos_y = n_pos_y
                arrow = n_arrow
                return 
            else:
                if not np.array_equal(n_obs['image'].transpose(1,0,2), obs['image'].transpose(1,0,2)):
                    n_inv = get_n_inv(args, n_obs, obs)
                else:
                    n_inv = inv
            front_obj = get_front_obj(args, env_id, world_map, pos_x, pos_y, arrow)
            write_log(args, save_path, f"The front object being interacted with is {front_obj}")
            obj_intr_rec[env_id][2][front_obj] += 1
            _, _, _ = update_world_map_view_step_memo_rec(args, env_id, world_map, pos_x, pos_y, arrow, n_obs, env_step_rec, env_memo_rec, env_view_rec, obj_view_rec)
            n_exp = get_exp(args, env_id, world_map, inv, act, obs, n_obs, o_world_map, n_inv, act_his, pos_x, pos_y, pos_x, pos_y, arrow, arrow)
            c_exp = sum_exp(args, n_exp, exp, act_his)

            exp = c_exp
            obs = n_obs
            inv = n_inv

        # We refresh the action history every args.refresh run to avoid too large action space
        if len(act_his) >= args.memo:
            act_his = act_his[1:]

    def get_action(self, env_id):
        self.action = generate_action(self.args, *self.log_action(), env_id)

        self.log(self.action)

    def get_desc(self, env_id):
        self.desc = generate_desc(self.args, *self.log_desc(env_id), env_id)
        
        self.log(self.desc)
    
    def get_n_exp(self, env_id):
        self.n_exp = generate_n_exp(self.args, *self.log_n_exp(env_id), env_id)

        self.log(self.n_exp)

    def get_reason(self, env_id):
        self.reason = generate_reason(self.args, *self.log_reason(), env_id)
        self.log(self.reason)

    def get_s_exp(self, env_id):
        self.exp = generate_s_exp(self.args, *self.log_s_exp(env_id), env_id)
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

    def old_world_map(self):
        o_world_map = {}
        o_world_map = self.world_map.copy()
        o_world_map[0] = self.world_map[0].copy()
        o_world_map[1] = self.world_map[1].copy()
        o_world_map[2] = self.world_map[2].copy()
        return o_world_map

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
        
        if self.args.all:
            self.envs = [i for i in range(len(env_ids))]

        # The self.pos_m stores the env position pos_x, pos_y, direction in the args.seed for all envs
        self.pos_m = get_pos_m(env_pos[str(self.args.seed)])

        for env_id in self.args.envs:
            self.save_path = get_path(self.args, env_id)
            # We need to determine the save path before print the configurations
            self.log(f"################## Starting Experiment ##################\n")
            self.log(f"Configurations are:\n{self.args}\n")
            # This self.exp is the experience particular for this env
            self.exp = train_exp(self.args, env_id, self.init_exp)
            self.inv = 0
            self.past_actions = []
            # The self.world_map, self.rec stores the world map for this particular env
            self.world_map, self.rec = get_track(env_id, env_sizes[str(self.args.seed)])

            if skip(env_id):
                continue
            
            # We obtain the starting position for this environment by using index [env_id]
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

                self.o_world_map = self.old_world_map()

                if isinstance(self.action, list):
                    img_l = []
                    metric_l = []
                    self.act_l = []
                    for act_int in self.action:
                        img_l.append(self.save_image(env_id, step))
                        metric_l.append(self.get_metrics())
                        act =int_act[str(act_int)]
                        self.act_l.append(act)
                        # World map updated inside the execute_action
                        self.execute_action(act, env_id)
                        if self.terminated:
                            break
                        step += 1
                    self.get_n_exp(env_id)
                    self.get_s_exp(env_id)
                    self.get_length()
                    for i in range(len(self.action)):
                        self.add_data(img_l[i], self.desc_user_1, self.desc, self.reason, str(self.action[i]), self.n_exp, self.s_exp)
                        self.add_metric(metric_l[i], self.length)

                elif isinstance(self.action, int):
                    act_s = act_obj[str(act)]
                    img = self.save_image(env_id, step)
                    self.execute_action(act_s)
                    self.get_metrics()
                    self.get_length
                    self.get_experience()
                    self.summarize_experience()
                    self.add_data(img, self.desc_user_1, self.desc, self.reason, str(self.action), self.n_exp, self.s_exp)
                    self.add_metric(self.metric, self.length)

            self.env_close()
            self.save_table()

        self.save_wandb()