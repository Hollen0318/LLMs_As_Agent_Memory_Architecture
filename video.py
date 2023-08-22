import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import re
from utils.load_video import *

def draw_text(draw, pos, content, size, width = None):
    draw.text(pos, content, fill = "black", font = ImageFont.truetype('arial.ttf', size))

def put_key(key, text):
    draw_text(draw, (pos[key][0], pos[key][1]), text, font_size[key])

def load_images_info(directory_path):
    images_info = []
    
    for filename in os.listdir(directory_path):
        if not filename.endswith(".png"):
            continue
        
        match = re.match(r"env_(\d+)_(\d+)_(\w+)_x_(\d+)_y_(\d+)_d_([\w_]+)\.png", filename)
        if match:
            env_id = int(match.group(1))
            image_id = int(match.group(2))
            act = match.group(3)
            x = int(match.group(4))
            y = int(match.group(5))
            direction = match.group(6).replace('_', '/')
            
            image_info = {
                'env_id': env_id,
                'image_id': image_id,
                'act': act,
                'x': x,
                'y': y,
                'direction': direction,
                'filename': filename
            }
            
            images_info.append(image_info)
    
    return sorted(images_info, key=lambda x: x['image_id'])

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
        "--envs",
        nargs = "+",
        help = "the range to load the env records from x to y",
        default = ["0", "0"]
    )
    parser.add_argument(
        "--fps",
        default = 10.0,
        type = float,
        help = "fps"
    )
    parser.add_argument(
        "--image-size",
        nargs = "+",
        help = "the size to create the video",
        default = ["2000", "1200"]
    )
    parser.add_argument(
        "--load",
        default = r"2023-08-20 13-55-25",
        type = str,
        help = "the location to load records, images"
    )
    parser.add_argument(
        "--video-name",
        default = "video.mp4",
        type = str,
        help = "the name to save the video"
    )
    parser.add_argument(
        "--video-size",
        nargs = "+",
        help = "the size to create the video",
        default = ["3000", "1800"]
    )
    args = parser.parse_args()
    images_data = load_images_info(args.load)
    out = cv2.VideoWriter(os.path.join(args.load, args.video_name), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (int(args.video_size[0]), int(args.video_size[1])))
    for image_data in images_data:
        env_id = image_data['env_id']
        image_id = image_data['image_id']
        act = image_data['act']
        x = image_data['x']
        y = image_data['y']
        direction = image_data['direction']
        filename = image_data['filename']

        print(f"Loading image index = {image_id}")

        frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
        image = Image.open(os.path.join(args.load, filename))
        image = image.resize((img_size["screenshot"][0], img_size["screenshot"][1]), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(frame)
        # Add text to the screenshot
        for caption in ["screenshot_caption", "representation_caption", "world_map_caption", "description_by_agent_caption", "reason_of_choices_caption", "past_actions_caption", "new_experience_obtained_caption", "summarized_experience_caption", "memory_record_caption"]:
            draw_text(draw, (pos[caption][0], pos[caption][1]), content[caption], font_size[caption])
        # Add image to it
        frame.paste(image, (pos["screenshot"][0], pos["screenshot"][1]))
        # Add static statistics to it
        # Position
        put_key("position", f"Position = ({x}, {y})")
        # Direction
        put_key("direction", f"Direction = {direction}")
        # Step index
        put_key("step_index", f"Step index = {image_id}")
        # Action
        put_key("action", f"Action = {act}")
        # Finishing decorating the rame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        image.close()
        out.write(frame)  # write the frame

    out.release()
    print(f"The video was successfully created at \n{str(os.path.join(args.load, args.video_name))}")