import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

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
        "--env-table",
        default = "env_table.csv",
        type = str,
        help = "the location to load the environment record csv"
    )
    parser.add_argument(
        "--image-dir",
        default = r"C:\Users\holle\OneDrive - Duke University\LLM_As_Agent\GPT\0\goal_False_gpt_3_lim_100_memo_30_refresh_6_seed_23_static_False_steps_1000_temp_0.9_view_7\2023-07-17 14-58-21",
        type = str,
        help = "the location to load screenshot image folder"
    )
    parser.add_argument(
        "--fps",
        default = 10.0,
        type = float,
        help = "fps"
    )
    parser.add_argument(
        "--metrics-table",
        default = "metrics.csv",
        type = str,
        help = "the location to load the object record csv"
    )
    parser.add_argument(
        "--obj-table",
        default = "obj_table.csv",
        type = str,
        help = "the location to load the object record csv"
    )
    parser.add_argument(
        "--scn-table",
        default = "scn_table.csv",
        type = str,
        help = "the location to load the screenshot record csv"
    )
    parser.add_argument(
        "--video-name",
        default = "video.mp4",
        type = str,
        help = "the location to save the video"
    )
    parser.add_argument(
        "--world-table",
        default = "world_map_table.csv",
        type = str,
        help = "the location to load the world map csv"
    )
    args = parser.parse_args()
    save_path = args.image_dir
    images = [i for i in os.listdir(save_path) if i.endswith(".png")]
    images.sort(key=lambda x:int(x.split('_')[3]))
    metrics_df = pd.read_csv(os.path.join(save_path, args.metrics_table))
    obj_table_df = pd.read_csv(os.path.join(save_path, args.obj_table))
    env_table_df = pd.read_csv(os.path.join(save_path, args.env_table))
    scn_table_df = pd.read_csv(os.path.join(save_path, args.scn_table))
    args.video_size = (3000, 1800)
    args.image_size = (1000, 600)
    out = cv2.VideoWriter(os.path.join(save_path, args.video_name), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, args.video_size)
    idx = 0
    for i in images:
        img_path = os.path.join(save_path, i)
        img = Image.open(img_path)
        img = img.resize(args.image_size, Image.LANCZOS)

        frame = Image.new('RGB', args.video_size, 'white')
        frame.paste(img, (0,0))

        # Add action being tabken
        draw = ImageDraw.Draw(frame)
        ratio_font_size = 40
        table_font_size = 30
        offset = 100
        draw.text((args.image_size[0] * 1.5, args.video_size[1] / 6), f"environment # {i.split('_')[1][0]} action {idx+1} {i.split('_')[4][:-4]}", fill='black', font = ImageFont.truetype('arial.ttf', 80))
        draw.text((0, offset + args.video_size[1] / 3), "1. environment view ratio {:.3f}%".format(metrics_df['env_view_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 50), str(env_table_df['Env_View'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 250), "2. environment memo ratio {:.3f}%".format(metrics_df['env_memo_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 300), str(env_table_df['Env_Memo'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 500), "3. environment step ratio {:.3f}%".format(metrics_df['env_step_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 550), str(env_table_df['Env_Step'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 750), "4. object view ratio {:.3f}%".format(metrics_df['obj_view_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 800), str(obj_table_df['Obj_View'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 850), "5. object intr ratio {:.3f}%".format(metrics_df['obj_intr_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
        draw.text((0, offset + args.video_size[1] / 3 + 900), str(obj_table_df['Obj_Intr'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', table_font_size))
        # draw.text((0, args.image_size[1] * 3.0), "5. object intr ratio {:.3f}%".format(metrics_df['obj_intr_ratio'][idx]), fill='black', font = ImageFont.truetype('arial.ttf', ratio_font_size))
        draw.text((args.video_size[0] / 3, args.image_size[1] + 50), f"Gained experience = ", fill='black', font = ImageFont.truetype('arial.ttf', 40))
        # # Use function
        wrapped_text = wrap_text(str(scn_table_df['C_exp'][idx]).replace("\n", ""), 130)
        # print(f"\n\n############################### the experience is ###############################\n\n{scn_table_df['C_exp'][idx]}")
        # Loop over wrapped text (each line)
        y0, dy = args.image_size[1] + 100, 50  # y0 - initial y value, dy - offset on y axis for new line
        for i, line in enumerate(wrapped_text):
            y = y0 + i*dy
            draw.text((args.video_size[0] / 3, y), line, fill='black', font = ImageFont.truetype('arial.ttf', 30))
        # Convert the image to BGR color (which is the format OpenCV uses)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        print(f"*********************************finishing the image # {str(idx)} *********************************")
        out.write(frame)  # write the frame
        img.close()  # close the image file
        if idx < 999:
            idx += 1
    out.release()
    print(f"The video was successfully created at \n{str(os.path.join(save_path, args.video_name))}")