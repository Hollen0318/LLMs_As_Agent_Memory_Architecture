import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def convert_images_to_video(image_folder, video_name, image_duration, video_size, font_size):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[3]))  # sort images by action index

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, 1.0 / image_duration, video_size)
    idx = 0
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = Image.open(img_path)
        img = img.resize(video_size, Image.LANCZOS)  # resize image

        # Add subtitle
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', font_size)
        draw.text((300, 100), image.split('_')[4][:-4], fill='white', font=font)
        draw.text((50, 100), f"Action {idx}", fill='white', font=font)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        out.write(frame)  # write the frame
        img.close()  # close the image file

        idx += 1

    # Release everything if job is finished
    out.release()
    print(f"The video was successfully created at \n\n{str(os.path.join(image_folder, video_name))}")

if __name__ == '__main__':
    address = r"C:\Users\holle\OneDrive - Duke University\LLM_As_Agent\GPT\0\goal_False_gpt_3_lim_800_memo_5_refresh_6_seed_23_static_False_steps_1000_temp_0.8_view_7\2023-07-15 13-33-25"
    # Example usage:
    convert_images_to_video(
        address,
        "video.mp4",
        0.8,
        (1280, 960),
        50
    )
