import os,json
import pdb
import glob
import cv2
import subprocess
import sys
import librosa
import numpy as np
from random import randint
import scipy.io.wavfile
from PIL import Image
import torchvision.transforms as transforms
#from  __builtin__ import any as b_any
train_data = json.loads(open("train/kinetics_train.json").read())

classes = ["blowing nose", "bowling", "chopping wood", "ripping paper", "shuffling cards", "singing", "tapping pen", "typing", "blowing out", "dribbling ball", "laughing", "mowing the lawn by pushing lawnmower","shoveling snow", "stomping", "tap dancing", "tapping guitar", "tickling","fingerpicking", "patting", "playing accordion", "playing bagpipes","playing bass guitar", "playing clarinet", "playing drums","playing guitar", "playing harmonica", "playing keyboard", "playing organ", "playing piano", "playing saxophone", "playing trombone","playing trumpet", "playing violin", "playing xylophone"]

path = "../videos/train/clipped"
audio_path = "../audio/train"

def extract_audio(path,audio_path):
	for vid in os.listdir(path):
		dest_path = os.path.join(audio_path,vid.strip('.mp4')+'.wav')
		src_path = os.path.join(path,vid)
		#print(src_path,dest_path)
		if not os.path.exists(dest_path):
			command = "ffmpeg -y -i %s -ac 1 -f wav %s" % (src_path, dest_path)
			os.system(command)
#extract_audio(path,audio_path)
#pdb.set_trace()

def upsample_audio(audio_path):
	for aud in os.listdir(audio_path):
		src_path = os.path.join(audio_path,aud)
		dest_path = os.path.join(audio_path,aud.strip('.wav')+'_48.wav')
		#ffmpeg -i tao-bD_Yi7Y.wav -ar 48000 tao-bD_Yi7Y_48.wav
		if not os.path.exists(dest_path):
			command = "ffmpeg -i %s -ar 48000 %s" %(src_path,dest_path)	
			os.system(command)
print("Upsampling Audios..........")
#upsample_audio(audio_path)

def make_data_tuple(path,audio_path):
	TUPLE = []
	time_length = 1
	fps=30
	frame_seq = 5
	frame_no = (frame_seq /(time_length*fps))
	startRead =0
	endRead = 1
	LABEL = []
	cou = 0
	for vid in os.listdir(path)[10000:15000]:
		vid_id = vid.strip('.mp4')
		try:
			tag = train_data[vid_id]['annotations']['label']
		#pdb.set_trace()
		except:
			tag = 'UNK'
			print('UNK encountered')
		src_path = os.path.join(path,vid)
		try:
			cap = cv2.VideoCapture(src_path)
		except:
			print("Unable to Video Capture")
			continue
		ret, vid_frame = cap.read()
		#Get Audio Frame
		aud_path = os.path.join(audio_path,vid.strip('.mp4')+'_48.wav')
		try:
			y, sr = librosa.load(aud_path)
		except:
			print("Unable to Audio Path")
			continue

		try:
			start_index = librosa.time_to_samples(0.0)
			end_index = librosa.time_to_samples(1.0)
			slice = y[int(start_index):int(end_index)]
			S = librosa.feature.melspectrogram(y=slice, sr=sr,n_mels=128)
			log_S = librosa.power_to_db(S**2, ref=np.max)
			S = np.resize(log_S,(128,128)) 		
			vid_frame = Image.fromarray(np.uint8(vid_frame*255))
		except:
			continue
		normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
		)
		preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		#transforms.Scale((224, 224)),
		transforms.ToTensor(),
		normalize
		])
		vid_frame = preprocess(vid_frame)
		#mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
		new_tup = [np.array(vid_frame),S]
		#pdb.set_trace()
		LABEL.append(tag)
		TUPLE.append(new_tup)
		cou+=1
		print(cou)
		if cou == 5000:
			np.save(open('5k_part3_dataset_trainX.npy','wb'),TUPLE)
			np.save(open('5k_part3_dataset_trainY.npy','wb'),np.asarray(LABEL))
		'''
		if cou == 10000:
			np.save(open('10k_part2_dataset_trainX.npy','wb'),TUPLE)
			np.save(open('10k_part2_dataset_trainY.npy','wb'),np.asarray(LABEL))
		'''	
			
		print(new_tup[0].shape,new_tup[1].shape)
	TUPLE = np.asarray(TUPLE)
	np.save(open('5k_part3_dataset_trainX.npy','wb'),TUPLE)
	np.save(open('5k_part3_dataset_trainY.npy','wb'),np.asarray(LABEL))
	print(len(os.listdir(path)),TUPLE.shape[0])	

print("Making Data Pairs.....................")

make_data_tuple(path,audio_path)

print(len(classes))
'''
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
'''	
