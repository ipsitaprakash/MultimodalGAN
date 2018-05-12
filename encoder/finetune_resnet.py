import torchvision.models as models
#from abstractions.dataset import Dataset
from sklearn.model_selection import train_test_split
#from utils import utils
#from skimage import io, transform
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

def create_splits():
	#Creating train test val split
	X = np.load("img_ft.npy")
	y = np.load("img_y.npy")
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify = y)
	# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test)
	# np.save("data/X_test",X_test)
	np.save("img_X_train",X_train)
	np.save("img_X_val",X_val)

	# np.save("data/y_test",y_test)
	np.save("img_y_train",y_train)
	np.save("img_y_val",y_val)

#preprocess = transforms.Compose([transforms.Resize(224),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), transforms.ToTensor()])
class my_loader(Dataset):
	def __init__(self, data_toload):
		global train_data_len
		if data_toload == "train":
			self.X, self.y = np.load("img_X_train.npy"), np.load("img_y_train.npy")
			train_data_len = self.X.shape[0]
		elif data_toload == "val":
			self.X, self.y = np.load("img_X_val.npy"), np.load("img_y_val.npy")
		elif data_toload == "test":
			self.X, self.y = np.load("data/X_test.npy"), np.load("data/y_test.npy")
		self.y = np.reshape(self.y,(self.y.shape[0],1))
		self.data_toload = data_toload
		self.data_path = "/pylon5/ir3l68p/madhurad/lsma/data/oxford_data/images/"
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
		img = Image.open(self.data_path+self.X[idx])
		img = self.preprocess(img)
		return img.float(), torch.from_numpy(np.array(self.y[idx])).long()

class forDataLoader(Dataset):
    def __init__(self, data_toload):
        if data_toload == "val":
            xData = np.load( "img_X_val.npy", encoding="bytes" )
            lab = np.load( "img_y_val.npy", encoding="bytes" )
        if data_toload == "train":
            xData = np.load( "img_X_train.npy", encoding="bytes" )
            lab = np.load( "img_y_train.npy", encoding="bytes" )
        idx = np.arange(xData.shape[0])
        np.random.shuffle(idx)

        self.X_train = xData[idx]
        self.y_train = lab[idx]

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        xD = self.X_train[index]
        xL = self.y_train[index]
        return xD, xL

def build_model(num_classes):	
	pt_model = models.resnet50(pretrained=True)
	ctr = 0
	for child in pt_model.children():
		if child.__class__.__name__ == 'Sequential':
			ctr+=1
		if ctr < 1:
			for param in child.parameters():
				param.requires_grad = False
	
	#num_ftrs = pt_model.fc.in_features
	#pt_model.fc = nn.Sequential(nn.Linear(num_ftrs, 1024), nn.LeakyReLU(), nn.Linear(1024, 128), nn.LeakyReLU(), nn.Linear(128, num_classes))
	# pt_model.fc = nn.Linear(num_ftrs, num_classes)
	return pt_model


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
                x = F.leaky_relu(self.l2(x))
                x = F.leaky_relu(self.l3(x))
                return x
#create_splits()
model = build_model(34)
model = ResNet_Modified(model)
if torch.cuda.is_available():
	print("Running in cuda!")
	model = model.cuda()
model.load_state_dict(torch.load("models_finetuned/model_res50_3.pkl"))
learning_rate = 0.000001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay = 1e-7)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 1e-7)
running_loss = 0
batch_size = 32
num_epochs = 45
train_data_len = 0
loss_ctr = []
mode = sys.argv[1]
if mode == "train":
	train_data = forDataLoader("train")
	val_data = forDataLoader("val")
	train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=4)
	val_loader = DataLoader(dataset=val_data,batch_size=batch_size,shuffle=False, num_workers=4)
	for epoch in range(4, num_epochs):
		start = time.time()
		count = 0
		running_loss = 0
		model.train()
		for i, (img, labels) in enumerate(train_loader):
			img = Variable(img)
			labels = Variable(labels).squeeze()
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
			img = Variable(img)
			labels = Variable(labels.squeeze())
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
		torch.save(model.state_dict(), 'models_finetuned/model_res50_'+str(epoch)+'.pkl')
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
	
