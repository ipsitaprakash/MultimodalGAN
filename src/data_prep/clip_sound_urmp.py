import os,json
import pdb
import glob
#import cv2
#import subprocess
#import sys
import librosa
import numpy as np
#from random import randint
#import scipy.io.wavfile



'''
audio_path = "/pylon5/ir3l68p/anuapk24/DL_project/audio/train"
path = "/pylon5/ir3l68p/anuapk24/DL_project/videos/train/clipped"
f = open("audio_files.txt","w")
for vid in os.listdir(path):
	aud_path = os.path.join(audio_path,vid.strip('.mp4')+'.wav')
	print(aud_path)
	clip_path = os.path.join(audio_path,vid.strip('.mp4')+'_clip.wav')
	f.write(clip_path+'\n')
	y, sr = librosa.load(aud_path,duration=1.0)
	librosa.output.write_wav(clip_path, y, sr)	


print("Done Clipping and Saving videos")

'''
f = open("urmp_audio_files.txt","w")
audio_path = "/pylon5/ir3l68p/anuapk24/DL_project/Sub-URMP/chunk/"
path = "/pylon5/ir3l68p/anuapk24/DL_project/videos/train/clipped"

for vid in os.listdir(audio_path + "validation"):
	for sound in os.listdir(audio_path +"validation/" + vid + "/")[:2000]:
		aud_path = os.path.join(audio_path,"validation",vid, sound)
		f.write(aud_path+'\n')

for vid in os.listdir(audio_path + "train"):
        for sound in os.listdir(audio_path + "train/" + vid + "/")[:2000]:
                aud_path = os.path.join(audio_path, "train", vid, sound)
                f.write(aud_path+'\n')
f.close()
print("Done converting to MP3")

