import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import re

def wrap_text(text, width):
    words = text.split(' ')
    lines = []
    line = ''

    for word in words:
        if len(line + ' ' + word) <= width:
            line += ' ' + word
        else:
            lines.append(line)
            line = word

    lines.append(line)
    return lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        default = r"output/seed_23/train/GPT/ENV 0 steps 40 gpt 0 temp 0.7 view 3/2023-08-20 12-41-44",
        type = str,
        help = "the location to load records, images"
    )
    parser.add_argument(
        "--fps",
        default = 10.0,
        type = float,
        help = "fps"
    )
    parser.add_argument(
        "--video-name",
        default = "video.mp4",
        type = str,
        help = "the name to save the video"
    )
    parser.add_argument(
        "--envs",
        nargs = "+",
        help = "the range to load the env records from x to y",
        default = ["0", "14"]
    )
    parser.add_argument(
        "--idx",
        nargs = "+",
        help = "the range to load the idx from x to y",
        default = ["0", "19"]
    )
    parser.add_argument(
        "--video-size",
        nargs = "+",
        help = "the size to create the video",
        default = ["3000", "1800"]
    )
    parser.add_argument(
        "--image-size",
        nargs = "+",
        help = "the size to create the video",
        default = ["1000", "600"]
    )
    parser.add_argument(
        "--config",
        type = str,
        help = "the configuration to show in the video",
        default = "desc_50_goal_False_gpt_3_lim_50_\nmemo_5_reason_50_refresh_6_seed_23_\nstatic_False_steps_20_temp_0.8_view_7"
    )
    args = parser.parse_args()
    save_path = args.load
    out = cv2.VideoWriter(os.path.join(save_path, args.video_name), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (int(args.video_size[0]), int(args.video_size[1])))
    for i  in range(int(args.envs[0]), int(args.envs[1])+1):
        print(f"Loading environment {i} start")
        scn_table_n = f"scn_table_env_{str(i)}.csv"
        rec_table_n = f"rec_table_env_{str(i)}.csv"
        metrics_table_n = f"metrics_env_{str(i)}.csv"
        length_table_n = f"length_table_env_{str(i)}.csv"
        world_map_table_n = f"world_map_table_env_{str(i)}.csv"
        # Image,Message,Reason,Action,N_exp,C_exp
        scn_table_df = pd.read_csv(os.path.join(save_path, scn_table_n))
        # Env_View, Env_Step, Env_Memo, Obj_View, Toggle, Pick Up, Drop Off
        rec_table_df = pd.read_csv(os.path.join(save_path, rec_table_n))
        # env_view_ratio,env_memo_ratio,env_step_ratio,obj_view_ratio,obj_intr_ratio,exp_length
        metrics_table_df = pd.read_csv(os.path.join(save_path, metrics_table_n))
        # World_Map_Object,World_Map_Color,World_Map_Status
        world_map_table_df = pd.read_csv(os.path.join(save_path, world_map_table_n))
        # Sum, desc_sys, desc_user_0, desc_assis, desc_user_1

        action = "start"
        image_n = f"env_{str(i)}_action_0_{action}.png"
        world_map_obj, world_map_col, world_map_sta = world_map_table_df['World_Map_Object'][0], world_map_table_df['World_Map_Object'][0], world_map_table_df['World_Map_Object'][0]
        pattern = re.compile(f'env_{str(i)}_idx_1_desc_\\d+\\.txt')
        message = scn_table_df["Message"][0]
        reason = scn_table_df["Reason"][0]
        image = Image.open(os.path.join(save_path, image_n))
        image = image.resize((int(args.image_size[0]), int(args.image_size[1])), Image.Resampling.LANCZOS)
        
        # Create the frame
        frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
        
        # Add image to it
        frame.paste(image, (0,0))
        
        # Add text to it
        draw = ImageDraw.Draw(frame)
        draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx 0",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

        # Finishing decorating the rame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame)  # write the frame

        # Update the action first
        # Create the frame
        frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
        
        # Add image to it
        frame.paste(image, (0,0))
        action = scn_table_df['Action'][0]
        # Add text to it
        draw = ImageDraw.Draw(frame)
        draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx 0",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 250), f"Choose to do {action}",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))

        draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

        # Finishing decorating the rame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame)  # write the frame

        # Create the frame
        frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
        
        # Add image to it
        frame.paste(image, (0,0))
        n_exp = scn_table_df['N_exp'][0]
        # Add text to it
        draw = ImageDraw.Draw(frame)
        draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx 0",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 250), f"Choose to do {action}",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 350), f"Get experience:",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 450), n_exp,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))

        draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

        # Finishing decorating the rame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame)  # write the frame

        # Create the frame
        frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
        
        # Add image to it
        frame.paste(image, (0,0))
        c_exp = scn_table_df['C_exp'][0]
        # Add text to it
        draw = ImageDraw.Draw(frame)
        draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx 0",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 250), f"Choose to do {action}",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 350), f"Get experience:",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 450), n_exp,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 550), f"Summarized experience:",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((500, int(args.image_size[1]) + 650), c_exp,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
        draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
        
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
        draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][0]), fill='black', font = ImageFont.truetype('arial.ttf', 20))

        draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

        # Finishing decorating the rame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame)  # write the frame

        image.close()
        for j in range(int(args.idx[0]), int(args.idx[1])):
            print(f"Loading environment {i} idx {j}")
            frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
            action = scn_table_df['Action'][j]
            message = scn_table_df["Message"][j+1]

            image_n = f"env_{str(i)}_idx_{str(j+1)}_act_{action}.png"
            image = Image.open(os.path.join(save_path, image_n))
            image = image.resize((int(args.image_size[0]), int(args.image_size[1])), Image.Resampling.LANCZOS)
            # Create the frame
            frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
            reason = scn_table_df["Reason"][j+1]
            # Add image to it
            frame.paste(image, (0,0))
            
            # Add text to it
            draw = ImageDraw.Draw(frame)
            draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx {str(j+1)}",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

            # Finishing decorating the rame
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame)  # write the frame

            # Update the action first
            # Create the frame
            frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
            
            # Add image to it
            frame.paste(image, (0,0))
            action = scn_table_df['Action'][j+1]
            # Add text to it
            draw = ImageDraw.Draw(frame)
            draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx {str(j+1)}",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 250), f"Choose to do {action}",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))

            draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

            # Finishing decorating the rame
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame)  # write the frame

            # Create the frame
            frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
            
            # Add image to it
            frame.paste(image, (0,0))
            n_exp = scn_table_df['N_exp'][j+1]
            # Add text to it
            draw = ImageDraw.Draw(frame)
            draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx {str(j+1)}",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 250), f"Choose to do {action}",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 350), f"Get experience:",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 450), n_exp,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))

            draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

            # Finishing decorating the rame
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame)  # write the frame

            # Create the frame
            frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
            
            # Add image to it
            frame.paste(image, (0,0))
            c_exp = scn_table_df['C_exp'][j+1]
            # Add text to it
            draw = ImageDraw.Draw(frame)
            draw.text((int(args.image_size[0]) + 100, 100), f"env id {str(i)} idx {str(j+1)}",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((500, int(args.image_size[1]) + 50), reason,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 250), f"Choose to do {action}",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 350), f"Get experience:",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 50), "environment view ratio {:.3f}%".format(metrics_table_df['env_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 100), "environment memo ratio {:.3f}%".format(metrics_table_df['env_memo_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 150), "environment step ratio {:.3f}%".format(metrics_table_df['env_step_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 200), "object view ratio {:.3f}%".format(metrics_table_df['obj_view_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 250), "object interact ratio {:.3f}%".format(metrics_table_df['obj_intr_ratio'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 300), "experience length {:.3f}%".format(metrics_table_df['exp_length'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 450), n_exp,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 550), f"Summarized experience:",  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((500, int(args.image_size[1]) + 650), c_exp,  fill='black', font = ImageFont.truetype('arial.ttf', 30))
            draw.text((0, int(args.image_size[1])), message,  fill='black', font = ImageFont.truetype('arial.ttf', 15))
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 350), "Environment View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 350), str(env_table_df['Env_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 600), "Environment Step Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 600), str(env_table_df['Env_Step'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 850), "Environment Status Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 850), str(env_table_df['Env_Memo'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 250, int(args.image_size[1]) + 1100), "Object Interact Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 500, int(args.image_size[1]) + 1100), str(obj_table_df['Obj_Intr'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))
            
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1100), "Object View Map", fill='black', font = ImageFont.truetype('arial.ttf', 20))
            draw.text((int(args.video_size[0]) - 900, int(args.image_size[1]) + 1150), str(obj_table_df['Obj_View'][j+1]), fill='black', font = ImageFont.truetype('arial.ttf', 20))

            draw.text((int(args.image_size[0]) + 100, 200), "Configurations:",  fill='black', font = ImageFont.truetype('arial.ttf', 80))
            draw.text((int(args.image_size[0]) + 100, 300), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

            # Finishing decorating the rame
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame)  # write the frame

            image.close()
    out.release()
    print(f"The video was successfully created at \n{str(os.path.join(save_path, args.video_name))}")