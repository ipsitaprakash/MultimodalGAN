import h5py
import torchvision.models as models
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
import pdb
import numpy as np
import sys, os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import os,json
import cv2
import subprocess
import sys
import pickle

class ResNet_Modified(nn.Module):
            def __init__(self, original_model):
                super(ResNet_Modified, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(original_model.children())[:-1]
                )
                num_ftrs = original_model.fc.in_features
                num_classes = 34
                self.l1 = nn.Linear(num_ftrs, 1024)
                self.l2 = nn.Linear(1024, 128)
                self.l3 = nn.Linear(128, num_classes)
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0),-1)
                x = F.leaky_relu(self.l1(x))
                return self.l2(x)
                """
                x = F.leaky_relu(self.l2(x))
                x = F.leaky_relu(self.l3(x))
                return x
                """

def build_model(num_classes):	
	pt_model = models.resnet50(pretrained=True)
	ctr = 0
	for child in pt_model.children():
		if child.__class__.__name__ == 'Sequential':
			ctr+=1
		if ctr < 1:
			for param in child.parameters():
				param.requires_grad = False
	return pt_model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_classes = 34
        num_ftrs = 800*64
        self.l1 = nn.Linear(num_ftrs, 1024)
        self.l2 = nn.Linear(1024, 128)
        self.l3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        #x = F.leaky_relu(self.l1(x))
        return self.l1(x)
        #return self.l2(x)

soundnet_model = Net()

resnet_model = build_model(34)
resnet_model = ResNet_Modified(resnet_model)
if torch.cuda.is_available():
	print("Running in cuda!")
	resnet_model.cuda()
	soundnet_model.cuda()

soundnet_model.load_state_dict(torch.load("aud_enc_models/6sound.pt"))
resnet_model.load_state_dict(torch.load("models_finetuned/model_res50_3.pkl"))

train_data = json.loads(open("train/kinetics_train.json").read())

classes_old = ["blowing nose", "bowling", "chopping wood", "ripping paper", "shuffling cards", "singing", "tapping pen", "typing", "blowing out", "dribbling ball", "laughing", "mowing the lawn by pushing lawnmower","shoveling snow", "stomping", "tap dancing", "tapping guitar", "tickling","fingerpicking", "patting", "playing accordion", "playing bagpipes","playing bass guitar", "playing clarinet", "playing drums","playing guitar", "playing harmonica", "playing keyboard", "playing organ", "playing piano", "playing saxophone", "playing trombone","playing trumpet", "playing violin", "playing xylophone"]
classes = ["typing",  "dribbling ball", "laughing", "playing bagpipes", "playing clarinet", "playing drums","playing guitar", "playing piano", "playing saxophone"]
video_path = "../videos/train/clipped/"
audio_path = "../audio/train"
soundnet_path = "/pylon5/ir3l68p/anuapk24/DL_project/Caffrey/SoundNet-tensorflow/output_kinetics/"
frame_path = "/pylon5/ir3l68p/anuapk24/DL_project/Caffrey/frames/"
normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
                ])

normalize_gan = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
                )
preprocess_gan = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                normalize_gan
                ])
img = []
sound = []
sound_embedding = []
img_embedding = []
cls_dict = []
file_idx = []
skip_count = 0
ctr = 0

for feature in os.listdir(soundnet_path):
	try:
		frame_arr = []
		if ctr%100 == 0:
			print(ctr)
		ctr+=1
		file_name = "_".join(feature.split("_")[:-2])
		cls = classes.index(train_data[file_name]["annotations"]["label"])
		cap = cv2.VideoCapture(video_path + file_name + ".mp4" )
		ret, vid_frame 	= cap.read()
		ret, vid_frame  = cap.read()
		int_ctr = 0
		while ret == False and int_ctr < 10:
			ret, vid_frame  = cap.read()
			int_ctr += 1
		if int_ctr >= 10:
			continue
		else:
			frame_ctr = 0
			frame_arr.append(vid_frame)
			while frame_ctr < 35:
				ret, vid_frame  = cap.read()
				if frame_ctr%5 == 0:
					frame_arr.append(vid_frame)
				frame_ctr += 1
#		cv2.imwrite(frame_path + file_name+ ".jpg", vid_frame )
		soundnet_data = np.load(soundnet_path + feature)[:800]
		soundnet_feature = soundnet_model(Variable(torch.from_numpy(soundnet_data).unsqueeze(0)).cuda()).cpu().data.numpy()
		for vid_frame in frame_arr:		
			vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
			vid_frame = Image.fromarray(np.uint8(vid_frame))
			vid_frame_gan = preprocess_gan(vid_frame).data.numpy()
			vid_frame = preprocess(vid_frame).data.numpy()
			resnet_feature = resnet_model(Variable(torch.from_numpy(vid_frame).unsqueeze(0)).cuda()).cpu().data.numpy()
			img_embedding.append(resnet_feature)
			sound_embedding.append(soundnet_feature)
			cls_dict.append(cls)
			img.append(vid_frame_gan)
			file_idx.append(file_name)
			sound.append(soundnet_data)
			
		#soundnet_data = np.load(soundnet_path + feature)[:800]
		#resnet_feature = resnet_model(Variable(torch.from_numpy(vid_frame).unsqueeze(0)).cuda()).cpu().data.numpy()
		#soundnet_feature = soundnet_model(Variable(torch.from_numpy(soundnet_data).unsqueeze(0)).cuda()).cpu().data.numpy()

		#file_idx.append(file_name)
		#img.append(vid_frame_gan)
		#sound_embedding.append(soundnet_feature)
		#cls_dict.append(cls)
		#img_embedding.append(resnet_feature)
		#sound.append(soundnet_data)
		
	except:
		skip_count+=1


f = h5py.File('dataset_reduced_1024.hdf5', mode='w')
f.create_dataset('img', data=np.array(img), dtype="float64")
f.create_dataset('sound_embeddings', data=np.array(sound_embedding))
f.create_dataset('img_embeddings', data=np.array(img_embedding))
f.create_dataset('class', data=np.array(cls_dict))
f.create_dataset('sound', data=np.array(sound))
f.close()
np.save("dataset_1024", np.array(file_idx))
print(len(img))

