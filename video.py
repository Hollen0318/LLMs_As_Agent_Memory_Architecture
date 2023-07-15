import cv2
import glob
import os
from moviepy.editor import *

# Get a list of all the image paths
image_folder = 'path_to_your_images'
images = glob.glob(os.path.join(image_folder, '*.png'))

# Sort the images by index
images.sort(key=lambda x: int(os.path.split(x)[-1].split('_')[2]))

# Frame rate (in FPS)
frame_rate = 0.5  # 0.5 FPS corresponds to 2 seconds per image

# Get size of the images
img = cv2.imread(images[0])
height, width, layers = img.shape
size = (width,height)

# Create a VideoWriter object
out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)

# Write each image to the video file
for image in images:
    img = cv2.imread(image)
    out.write(img)
out.release()

# For adding subtitles and environment id at the top of the video
video = VideoFileClip('project.mp4')
# Assuming the environment id is static and present in the image name
env_id = images[0].split('_')[1]

# Add environment id as text clip at the top of the video
txt_clip = TextClip(f"Environment ID: {env_id}", fontsize = 70, color = 'white')
txt_clip = txt_clip.set_pos('top').set_duration(video.duration)

# Make a list of (start_second, end_second, text) tuples for subtitles
subtitles = [(i/frame_rate, (i+1)/frame_rate, os.path.split(img)[-1].split('_')[3].split('.')[0]) for i, img in enumerate(images)]

# Generate a subtitle clip
subs = SubtitlesClip(subtitles)

# Overlay the text clip and the subtitle clip onto the video
final = CompositeVideoClip([video, txt_clip.set_start(0), subs.set_start(0)])

final.write_videofile("final_output.mp4", codec='libx264')
