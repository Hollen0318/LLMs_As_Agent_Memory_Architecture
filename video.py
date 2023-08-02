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
        default = r"/home/hz271/Research/General Robotics Lab 通用机器人实验室/LLM_As_Agent/GPT/ALL/desc_50_goal_False_gpt_3_lim_50_memo_5_reason_50_refresh_6_seed_23_static_False_steps_20_temp_0.8_view_7/2023-08-02 12-14-31",
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
        default = ["0", "2"]
    )
    parser.add_argument(
        "--idx",
        nargs = "+",
        help = "the range to load the idx from x to y",
        default = ["0", "20"]
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
        default = "desc_50_goal_False_gpt_3_lim_50_memo_5_reason_50_refresh_6_seed_23_static_False_steps_20_temp_0.8_view_7"
    )
    args = parser.parse_args()
    save_path = args.load
    out = cv2.VideoWriter(os.path.join(save_path, args.video_name), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (int(args.video_size[0]), int(args.video_size[1])))
    for i  in range(int(args.envs[0]), int(args.envs[1])+1):
        scn_table_n = f"scn_table_env_{str(i)}.csv"
        obj_table_n = f"obj_table_env_{str(i)}.csv"
        metrics_table_n = f"metrics_env_{str(i)}.csv"
        world_map_table_n = f"world_map_table_env_{str(i)}.csv"
        env_table_n = f"env_table_env_{str(i)}.csv"
        # Image,Message,Reason,Action,N_exp,C_exp
        scn_table_df = pd.read_csv(os.path.join(save_path, scn_table_n))
        # Obj_Intr,Obj_View
        obj_table_df = pd.read_csv(os.path.join(save_path, obj_table_n))
        # env_view_ratio,env_memo_ratio,env_step_ratio,obj_view_ratio,obj_intr_ratio,exp_length
        metrics_table_df = pd.read_csv(os.path.join(save_path, metrics_table_n))
        # World_Map_Object,World_Map_Color,World_Map_Status
        world_map_table_df = pd.read_csv(os.path.join(save_path, world_map_table_n))
        # Env_View,Env_Step,Env_Memo
        env_table_df = pd.read_csv(os.path.join(save_path, env_table_n))

        action = "start"
        image_n = f"env_{str(i)}_action_0_{action}.png"
        world_map_obj, world_map_col, world_map_sta = world_map_table_df['World_Map_Object'][0], world_map_table_df['World_Map_Object'][0], world_map_table_df['World_Map_Object'][0]
        pattern = re.compile(f'env_{str(i)}_idx_1_desc_\\d+\\.txt')
        for filename in os.listdir(save_path):
            if pattern.match(filename):
                full_path = os.path.join(save_path, filename)
                with open(full_path, 'r') as file:
                    desc = file.read()
        message = scn_table_df["Message"][0]
        image = Image.open(os.path.join(save_path, image_n))
        image = image.resize((int(args.image_size[0]), int(args.image_size[1])), Image.Resampling.LANCZOS)
        
        # Create the frame
        frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
        
        # Add image to it
        frame.paste(image, (0,0))
        
        # Add text to it
        draw = ImageDraw.Draw(frame)
        draw.text((int(args.image_size[0]) * 1.5, int(args.video_size[1]) / 6), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

        # Finishing decorating the rame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame)  # write the frame
        # print(f"The message is \n{message}")
        # print(f"For the env = {i}, the row_id = 0, the action is {action}, the desc is {desc}")
        # print(f"the world map obj = {world_map_obj} col = {world_map_col} sta = {world_map_sta}")
        for j in range(int(args.idx[0]), int(args.idx[1])):
            frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
            action = scn_table_df['Action'][j]
            image_n = f"env_{str(i)}_idx_{str(j+1)}_act_{action}.png"
            image = Image.open(os.path.join(save_path, image_n))
            image = image.resize((int(args.image_size[0]), int(args.image_size[1])), Image.Resampling.LANCZOS)
            frame.paste(image, (0,0))

            # Add text to it
            draw = ImageDraw.Draw(frame)
            draw.text((int(args.image_size[0]) * 1.5, int(args.video_size[1]) / 6), args.config,  fill='black', font = ImageFont.truetype('arial.ttf', 80))

            world_map_obj, world_map_col, world_map_sta = world_map_table_df['World_Map_Object'][j+1], world_map_table_df['World_Map_Object'][j+1], world_map_table_df['World_Map_Object'][j+1]
            pattern = re.compile(f'env_{str(i)}_idx_{str(j+2)}_desc_\\d+\\.txt')
            for filename in os.listdir(save_path):
                if pattern.match(filename):
                    full_path = os.path.join(save_path, filename)
                    with open(full_path, 'r') as file:
                        desc = file.read()
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame)  # write the frame
    out.release()
    print(f"The video was successfully created at \n{str(os.path.join(save_path, args.video_name))}")
            # print(f"For the env = {i}, the row_id = {j+1}, the action is {action}, the desc is {desc}")
            # print(f"the world map obj = {world_map_obj} col = {world_map_col} sta = {world_map_sta}")
    # images = [i for i in os.listdir(save_path) if i.endswith(".png")]
    # images.sort(key=lambda x:int(x.split('_')[3]))
    # metrics_df = pd.read_csv(os.path.join(save_path, args.metrics_table))
    # obj_table_df = pd.read_csv(os.path.join(save_path, args.obj_table))
    # env_table_df = pd.read_csv(os.path.join(save_path, args.env_table))
    # scn_table_df = pd.read_csv(os.path.join(save_path, args.scn_table))
    # args.video_size = (3000, 1800)
    # args.image_size = (1000, 600)
    # out = cv2.VideoWriter(os.path.join(save_path, args.video_name), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, args.video_size)
    # idx = 0
    # for i in images:
    #     img_path = os.path.join(save_path, i)
    #     img = Image.open(img_path)
    #     img = img.resize(args.image_size, Image.LANCZOS)

    #     frame = Image.new('RGB', args.video_size, 'white')
    #     frame.paste(img, (0,0))

    #     # Add action being tabken
    #     draw = ImageDraw.Draw(frame)
    #     ratio_font_size = 40
    #     table_font_size = 30
    #     offset = 100
    #     draw.text((args.image_size[0] * 1.5, args.video_size[1] / 6), f"environment # {i.split('_')[1][0]} action {idx+1} {i.split('_')[4][:-4]}", fill='black', font = ImageFont.truetype('arial.ttf', 80))
    #     draw.text((0, offset + args.video_size[1] / 3), "1. environment view ratio {:.3f}%".format(metrics_df['env_view_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 50), str(env_table_df['Env_View'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 250), "2. environment memo ratio {:.3f}%".format(metrics_df['env_memo_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 300), str(env_table_df['Env_Memo'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 500), "3. environment step ratio {:.3f}%".format(metrics_df['env_step_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 550), str(env_table_df['Env_Step'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 750), "4. object view ratio {:.3f}%".format(metrics_df['obj_view_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 800), str(obj_table_df['Obj_View'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 850), "5. object intr ratio {:.3f}%".format(metrics_df['obj_intr_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
    #     draw.text((0, offset + args.video_size[1] / 3 + 900), str(obj_table_df['Obj_Intr'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
    #     # draw.text((0, args.image_size[1] * 3.0), "5. object intr ratio {:.3f}%".format(metrics_df['obj_intr_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
    #     draw.text((args.video_size[0] / 3, args.image_size[1] + 50), f"Gained experience = ", fill='black', font = ImageFont.truetype('arial.ttf', 40))
    #     # # Use function
    #     wrapped_text = wrap_text(str(scn_table_df['C_exp'][idx]).replace("\n", ""), 130)
    #     # print(f"\n\n############################### the experience is ###############################\n\n{scn_table_df['C_exp'][idx]}")
    #     # Loop over wrapped text (each line)
    #     y0, dy = args.image_size[1] + 100, 50  # y0 - initial y value, dy - offset on y axis for new line
    #     for i, line in enumerate(wrapped_text):
    #         y = y0 + i*dy
    #         draw.text((args.video_size[0] / 3, y), line, fill='black', font = ImageFont.truetype('arial.ttf', 30))
    #     # Convert the image to BGR color (which is the format OpenCV uses)
    #     frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    #     print(f"*********************************finishing the image # {str(idx)} *********************************")
    #     out.write(frame)  # write the frame
    #     img.close()  # close the image file
    #     if idx < 999:
    #         idx += 1
    # out.release()
    # print(f"The video was successfully created at \n{str(os.path.join(save_path, args.video_name))}")