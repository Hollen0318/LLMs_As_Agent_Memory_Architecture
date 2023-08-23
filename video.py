import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import re
from utils.load_video import *

def draw_text(draw, pos, content, size, width = None):
    draw.text(pos, content, fill = "black", font = ImageFont.truetype('arial.ttf', size))

def extract_inventory(s: str) -> int:
    # Correspondence relationship
    inventory_map = {
        "nothing": "0",
        "Empty": "1",
        "Wall": "2",
        "Floor": "3",
        "Door": "4",
        "Key": "5",
        "Ball": "6",
        "Box": "7",
        "Portal": "8",
        "Lava": "9"
    }

    # Extract the inventory line from the string
    inventory_line = [line for line in s.split('\n') if "Inventory" in line][0]

    # Extract the actual inventory item
    inventory_item = inventory_line.split("You are holding ")[1].strip()

    # Convert the item to its integer representation
    return inventory_item

def extract_past_actions(s: str) -> list:
    # Find the pattern for past actions using regex
    match = re.search(r'3\. Past actions (.+?)\n', s)
    if match:
        actions = match.group(1).split(', ')
        return actions
    else:
        return []


def load_images_info(directory_path):
    images_info = []
    
    for filename in os.listdir(directory_path):
        if not filename.endswith(".png"):
            continue
        
        # Updated regex pattern to handle spaces in 'act'
        match = re.match(r"env_(\d+)_(\d+)_(\w+\s*\w*)_x_(\d+)_y_(\d+)_d_([\w_]+)\.png", filename)
        
        if match:
            env_id = int(match.group(1))
            image_id = int(match.group(2))
            act = match.group(3)   # This will now capture "pick up" and "drop off" as well
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
    
    return sorted(images_info, key=lambda x: x['image_id'])

def put_key(key, text):
    draw_text(draw, (pos[key][0], pos[key][1]), text, font_size[key])

def put_key_wrap(key, text):
    text_wrapped = wrap_text(text, line_size[key])
    text = "\n".join(text_wrapped)
    draw_text(draw, (pos[key][0], pos[key][1]), text, font_size[key])

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
        default = r"2023-08-20 23-23-41",
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
    env_id = images_data[0]["env_id"]
    rec_table, scn_table, world_map_table, length_table, metrics_table = load_tables(args.load)
    choice_id = 0
    old_reason = scn_table.iloc[0]["reason"]
    new_reason = scn_table.iloc[0]["reason"]
    for image_data in images_data:
        env_id = image_data['env_id']
        image_id = image_data['image_id']
        act = image_data['act']
        x = image_data['x']
        y = image_data['y']
        direction = image_data['direction']
        filename = image_data['filename']
        print(f"Loading image index = {image_id} choice_id = {choice_id}")

        frame = Image.new('RGB', (int(args.video_size[0]), int(args.video_size[1])), 'white')
        image = Image.open(os.path.join(args.load, filename))
        image = image.resize((img_size["screenshot"][0], img_size["screenshot"][1]), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(frame)
        # Add text to the screenshot
        for caption in ["screenshot_caption", "world_map_caption", "description_by_agent_caption", "reason_of_choices_caption", "past_actions_caption", "new_experience_obtained_caption", "summarized_experience_caption", "memory_record_caption", "environment_view_caption", "environment_step_caption", "configuration_caption", "world_map_obj_caption", "metrics_caption"]:
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
        # Choice index
        put_key("choice_id", f"Choice index = {choice_id}")
        # Action
        put_key("action", f"Action = {act}")
        # Description
        put_key_wrap("description_by_agent", scn_table.iloc[image_id]["desc"])
        description_length = length_table.iloc[choice_id]["desc"]
        # Description Length
        put_key("description_length", f"Length: {description_length} Limit: 50")
        # New Experience Obtained
        put_key_wrap("new_experience_obtained", scn_table.iloc[image_id]["n_exp"])
        # New Experience Length
        new_experience_length = length_table.iloc[choice_id]["n_exp"]
        put_key("new_experience_obtained_length", f"Length: {new_experience_length} Limit: 50")
        # Summarized Experience
        put_key_wrap("summarized_experience", scn_table.iloc[image_id]["s_exp"])
        # Summarized Experience Length
        summarize_experience_length = length_table.iloc[choice_id][-1]
        put_key("summarized_experience_length", f"Length: {summarize_experience_length} Limit: 100")
        # World Map
        put_key_wrap("world_map", world_map_table.iloc[image_id]["c_world_map"])
        # Env View
        put_key("environment_view", rec_table.iloc[image_id]["env_view"])
        # Reason of Choices
        put_key_wrap("reason_of_choices", scn_table.iloc[image_id]["reason"])
        # Reason of Choices length
        reason_length = length_table.iloc[choice_id]["reason"]
        put_key("reason_of_choices_length", f"Length: {reason_length} Limit: 100")
        # Environment Step
        put_key("environment_step", rec_table.iloc[image_id]["env_step"])
        # Past Actions
        past_actions = extract_past_actions(scn_table.iloc[image_id]["obs"])
        past_actions_s = ", ".join(past_actions)
        put_key_wrap("past_actions", past_actions_s)
        # Inventory
        inventory = extract_inventory(scn_table.iloc[image_id]["obs"])
        put_key("inventory", f"Inventory: {inventory}")
        # Memory record
        put_key("memory_record", rec_table.iloc[image_id]["env_memo"])
        # World Map Object
        put_key("world_map_obj", world_map_table.iloc[image_id]["world_map_obj"])
        # Memory Setting
        put_key("memory_setting", f"Memory Length Limit = 10")
        # view ratio
        view_ratio = str(metrics_table.iloc[image_id]["env_view"])
        put_key("view_ratio", f"Environment View % = {view_ratio}")
        # step ratio
        step_ratio = str(metrics_table.iloc[image_id]["env_step"])
        put_key("step_ratio", f"Environment Step % = {step_ratio}")
        # toggle ratio
        toggle = str(metrics_table.iloc[image_id]["toggle"])
        put_key("toggle", f"Object Toggle % = {toggle}")
        # Pick up
        pick_up = str(metrics_table.iloc[image_id]["pick up"])
        put_key("pick up", f"Object Pick Up % = {pick_up}")
        # Drop off
        drop_off = str(metrics_table.iloc[image_id]["drop off"])
        put_key("drop off", f"Object Drop Off % = {drop_off}")
        # Finishing decorating the rame
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        image.close()
        out.write(frame)  # write the frame
        new_reason = scn_table.iloc[image_id]["reason"]
        if new_reason == old_reason:
            pass
        else:
            choice_id += 1
            old_reason = new_reason
    out.release()
    print(f"The video was successfully created at \n{str(os.path.join(args.load, args.video_name))}")