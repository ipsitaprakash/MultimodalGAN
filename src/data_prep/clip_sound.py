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
from pydub import AudioSegment



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
f = open("audio_files.txt","w")
audio_path = "/pylon5/ir3l68p/anuapk24/DL_project/audio/train"
path = "/pylon5/ir3l68p/anuapk24/DL_project/videos/train/clipped"

for vid in os.listdir(path):
	
	aud_path = os.path.join(audio_path,vid.strip('.mp4')+'.wav')
	#wav_audio = AudioSegment.from_file(aud_path, format="wav")
	mp3_path = os.path.join(audio_path,vid.strip('.mp4')+'_clip.mp3')
	#wav_audio.export(mp3_path, format="mp3")
	#print(mp3_path)
	f.write(aud_path+'\n')
print("Done converting to MP3")

