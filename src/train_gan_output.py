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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import h5py

def create_splits():
	#Creating train test val split
	pdb.set_trace()
	ds = h5py.File('MultimodalGAN/saved_frames/generator_samples.hdf5', mode='r')
	sample_shape = ds['img'].shape[0]
	X = []
	y = []
	for i in range(sample_shape):
		X.append(ds['img'][i])
		y.append(ds['class'][i])
	X = np.array(X)
	y = np.array(y)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify = y)
	# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test)
	# np.save("data/X_test",X_test)
	np.save("gan_X_train",X_train)
	np.save("gan_X_val",X_val)

	# np.save("data/y_test",y_test)
	np.save("gan_y_train",y_train)
	np.save("gan_y_val",y_val)

#preprocess = transforms.Compose([transforms.Resize(224),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), transforms.ToTensor()])
class my_loader(Dataset):
	def __init__(self, data_toload):
		global train_data_len
		if data_toload == "train":
			self.X, self.y = np.load("gan_X_train.npy"), np.load("gan_y_train.npy")
			train_data_len = self.X.shape[0]
		elif data_toload == "val":
			self.X, self.y = np.load("gan_X_val.npy"), np.load("gan_y_val.npy")
		self.y = np.reshape(self.y,(self.y.shape[0],1))
		self.data_toload = data_toload
		self.reshape = 224
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

		self.preprocess = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize
				])

	def __len__(self):
		return self.X.shape[0] - 1 

	def __getitem__(self,idx):
		return self.X[idx], self.y[idx]


class ImgClassifier(nn.Module):
    def __init__(self):
        super(ImgClassifier, self).__init__()
        self.conv1 = nn.Conv2d( in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1 )
        self.conv2 = nn.Conv2d( in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 )
        self.conv3 = nn.Conv2d( in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1 )
        self.conv4 = nn.Conv2d( in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1 )
        self.conv5 = nn.Conv2d( in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1 )
        self.conv6 = nn.Conv2d( in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 )
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.lin1 = nn.Linear(16384, 1024)
        self.lin2 = nn.Linear( 1024, 128 )
        self.lin3 = nn.Linear(128,9)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(self.conv2(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu( self.bn4(self.conv4( x ) ))
        x = F.leaky_relu( self.bn5(self.conv5( x ) ))
        x = self.conv6(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(x)
        x = self.lin3(x)
        return x

#create_splits()
model = ImgClassifier()
if torch.cuda.is_available():
	print("Running in cuda!")
	model = model.cuda()
#model.load_state_dict(torch.load("models_finetuned/model_res50_3.pkl"))
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay = 1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 1e-7)
running_loss = 0
batch_size = 32
num_epochs = 20
train_data_len = 0
loss_ctr = []
mode = sys.argv[1]
if mode == "train":
	train_data = my_loader("train")
	val_data = my_loader("val")
	train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=4)
	val_loader = DataLoader(dataset=val_data,batch_size=batch_size,shuffle=False, num_workers=4)
	for epoch in range(num_epochs):
		start = time.time()
		count = 0
		running_loss = 0
		model.train()
		for i, (img, labels) in enumerate(train_loader):
			img = Variable(img).float()
			labels = Variable(labels).long().squeeze()
			if torch.cuda.is_available():
				img = img.cuda()
				labels = labels.cuda()
			optimizer.zero_grad()  # zero the gradient buffer
			outputs = model(img)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.data[0]
			count+=1
			if (i+1) % 100 == 0:
				print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%(epoch+1, num_epochs, i+1, train_data_len//batch_size, loss.data[0]))
		print(" Running loss:", running_loss/count)
		total = 0.0
		correct = 0.0
		count_val = 0
		val_loss = 0.0
		model.eval()
		for img, labels in val_loader:
			img = Variable(img).float()
			labels = Variable(labels.squeeze()).long()
			if torch.cuda.is_available():
				img = img.cuda()
				labels = labels.cuda()
			outputs = model(img)
			loss = criterion(outputs, labels)
			val_loss += loss.data[0]
			count_val += 1
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted.cpu() == labels.data.cpu()).sum()
		print('Accuracy of the network on the frames: %f %%' % (100 * correct / total))
		print("Val loss: ",val_loss/count_val)
		print("Time taken for epoch:",time.time()-start)
#		loss_ctr.append([running_loss/count,val_loss/count_val])
#		torch.save(model.state_dict(), 'models_finetuned/model_res50_'+str(epoch)+'.pkl')
#		np.save("loss_res50_ctr_oxford",np.array(loss_ctr))

if mode == "test":
	test_data = forDataLoader("test")
	test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,num_workers=4)
	total = 0.0
	correct = 0.0
	model.eval()
	for img, labels in test_loader:
		img = Variable(img)
		labels = labels.squeeze()
		if torch.cuda.is_available():
			img = img.cuda()
		outputs = model(img)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted.cpu() == labels).sum()
	print('Accuracy of the network on the frames: %f %%' % (100 * correct / total))
	
