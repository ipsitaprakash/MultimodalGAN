import pdb
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.utils.data import Dataset
import torch.optim as optim

classes = ["UNK","blowing nose", "bowling", "chopping wood", "ripping paper", "shuffling cards", "singing", "tapping pen", "typing", "blowing out", "dribbling ball", "laughing", "mowing the lawn by pushing lawnmower","shoveling snow", "stomping", "tap dancing", "tapping guitar", "tickling","fingerpicking", "patting", "playing accordion", "playing bagpipes","playing bass guitar", "playing clarinet", "playing drums","playing guitar", "playing harmonica", "playing keyboard", "playing organ", "playing piano", "playing saxophone", "playing trombone","playing trumpet", "playing violin", "playing xylophone"] 

class forDataLoader(Dataset):
    def __init__(self):
        xData = numpy.load( "med_dataset_trainX.npy", encoding="bytes" )
        lab = numpy.load( "med_dataset_trainY.npy", encoding="bytes" )
        self.X_train = xData
        self.y_train = lab

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        xD = self.X_train[index]
        xL = self.y_train[index]
        return xD, xL

def myCustomCollate(batch):
    return batch[0 ][0 ], batch[0 ][1 ]

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=8, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*199*257,1024)
        self.fc2 = nn.Linear(1024,128)
        self.fc3 = nn.Linear(128,35)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(-1,16*199*257)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

dSet = forDataLoader()
bS = 1
train_loader = torch.utils.data.DataLoader(dSet, batch_size=bS, collate_fn=myCustomCollate, num_workers=2)
net = Net()

classes = ["UNK", "blowing nose", "bowling", "chopping wood", "ripping paper", "shuffling cards", "singing", "tapping pen", "typing", "blowing out", "dribbling ball", "laughing", "mowing the lawn by pushing lawnmower","shoveling snow", "stomping", "tap dancing", "tapping guitar", "tickling","fingerpicking", "patting", "playing accordion", "playing bagpipes","playing bass guitar", "playing clarinet", "playing drums","playing guitar", "playing harmonica", "playing keyboard", "playing organ", "playing piano", "playing saxophone", "playing trombone","playing trumpet", "playing violin", "playing xylophone"]
labelDict = dict()

cnt = 0
for c in classes:
    labelDict[c] = cnt
    cnt += 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

ep = 15
for e in range(ep):
    for i, d in enumerate( train_loader ):

        v = Variable(torch.from_numpy(d[0][0]).float())
        a = Variable(torch.from_numpy(d[0][1]).float())
        lab = d[1]
        y = labelDict[lab]
        y = Variable(torch.from_numpy(numpy.asarray(y).reshape((1))))

        a = a.unsqueeze(0)
        a = a.unsqueeze(0)

        if torch.cuda.is_available():
            net = net.cuda()
            a = a.cuda()
            y = y.cuda()

        out = net(a)
        optimizer.zero_grad()
        loss = criterion( out, y )
        
        print(loss.data[0])
        avgLoss += loss.data
        cnt += 1

        loss.backward()
        optimizer.step()

    print("avgLoss")
    print(avgLoss/cnt)

    torch.save(net.state_dict(), "aud_models/"+str(e)+"sound.pt")

