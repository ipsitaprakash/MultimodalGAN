from sklearn.model_selection import train_test_split
import pdb
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import numpy as np
from torch.utils.data import Dataset
import torch.optim as optim

def create_splits():
        #Creating train test val split
        X = np.load("soundnet_ft.npy")
        y = np.load("soundnet_y.npy")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify = y)
        # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test)
        # np.save("data/X_test",X_test)
        np.save("soundet_X_train",X_train)
        np.save("soundnet_X_val",X_val)

        # np.save("data/y_test",y_test)
        np.save("soundet_y_train",y_train)
        np.save("soundnet_y_val",y_val)

class forDataLoader(Dataset):
    def __init__(self, data_toload):
        if data_toload == "val":
            xData = np.load( "soundnet_X_val.npy", encoding="bytes" )
            lab = np.load( "soundnet_y_val.npy", encoding="bytes" )
        if data_toload == "train":
            xData = np.load( "soundet_X_train.npy", encoding="bytes" )
            lab = np.load( "soundet_y_train.npy", encoding="bytes" )
        idx = np.arange(xData.shape[0])
        print(xData.shape[0])
        np.random.shuffle(idx)
        self.X_train = xData[idx]
        self.y_train = lab[idx]
    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        xD = self.X_train[index]
        xD = np.load(xD)[:800]
        xL = self.y_train[index]
        return xD, xL

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
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        return x

#create_splits()
dSet1 = forDataLoader("train")
bS = 32
train_loader = torch.utils.data.DataLoader(dSet1, batch_size=bS, num_workers=4, shuffle=True)
dSet2 = forDataLoader("val")
bS = 32
val_loader = torch.utils.data.DataLoader(dSet2, batch_size=bS, num_workers=2, shuffle=False)
net = Net()
net.cuda()
labelDict = dict()

cnt = 0
net.load_state_dict(torch.load("aud_enc_models/6sound.pt"))
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.Adam(net.parameters(), lr=0.000001, weight_decay=1e-7)
ep = 15
for e in range(7,ep):
    avgLoss = 0
    cnt = 0
    for i, d in enumerate( train_loader ):
        v = Variable(d[0]).float()
        y = Variable(d[1]).long()
        #v = v.unsqueeze(1)
        #a = a.unsqueeze(0)
        #a = a.unsqueeze(0)
        v  = v.permute(0,2,1)
        if torch.cuda.is_available():
            net = net.cuda()
            v = v.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        out = net(v)
        loss = criterion( out, y )
        if i%100 == 0: 
            print(i,":",loss.item())
        avgLoss += loss.item()
        cnt += 1
        loss.backward()
        optimizer.step()

    print("Epoch:",e,"avgLoss:", avgLoss/cnt)
    torch.save(net.state_dict(), "aud_enc_models/"+str(e)+"sound.pt")
    correct = 0.0
    total = 0.0
    for frames, labels in val_loader:
        frames = Variable(frames.float())
        #frames = frames.unsqueeze(1)
        frames  = frames.permute(0,2,1).cuda()
        outputs = net(frames)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum() 
    print('Accuracy of the network on the frames: %f %%' % (100 * correct / float(total)))
