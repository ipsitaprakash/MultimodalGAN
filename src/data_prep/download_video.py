import os
import json
import pdb
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

classes = ["blowing nose", "bowling", "chopping wood", "ripping paper", "shuffling cards", "singing", "tapping pen", "typing", "blowing out", "dribbling ball", "laughing", "mowing the lawn by pushing lawnmower","shoveling snow", "stomping", "tap dancing", "tapping guitar", "tickling","fingerpicking", "patting", "playing accordion", "playing bagpipes","playing bass guitar", "playing clarinet", "playing drums","playing guitar", "playing harmonica", "playing keyboard", "playing organ", "playing piano", "playing saxophone", "playing trombone","playing trumpet", "playing violin", "playing xylophone"]

train_data = json.loads(open("train/kinetics_train.json").read())
"""
for vid_id in train_data.keys():
	if train_data[vid_id]['annotations']['label'] in classes:
		if not os.path.exists("/pylon5/ir3l68p/anuapk24/DL_project/videos/train/" + vid_id + ".mp4"):
			cmd = "youtube-dl "+ train_data[vid_id]['url'] + " -f mp4 -o  /pylon5/ir3l68p/anuapk24/DL_project/videos/train/" + vid_id + ".mp4"
			print (cmd)
			os.system(cmd)
"""
count = 0
print(len(train_data.keys()))
for vid_id in train_data.keys():
		
	if train_data[vid_id]['annotations']['label'] in classes:
		1
		if os.path.exists("/pylon5/ir3l68p/anuapk24/DL_project/videos/train/" + vid_id + ".mp4"):
			#count+=1
			if not os.path.exists("/pylon5/ir3l68p/anuapk24/DL_project/videos/train/clipped/" + vid_id + ".mp4"):
				count+=1
				ffmpeg_extract_subclip( "/pylon5/ir3l68p/anuapk24/DL_project/videos/train/" + vid_id + ".mp4", int(train_data[vid_id]['annotations']['segment'][0]), int(train_data[vid_id]['annotations']['segment'][1]), targetname="/pylon5/ir3l68p/anuapk24/DL_project/videos/train/clipped/" + vid_id + ".mp4")

print(count)
