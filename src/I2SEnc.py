from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.utils.data import Dataset
import torch.optim as optim

class forDataLoader(Dataset):
    def __init__(self):
        xData = numpy.load( "small_dataset_trainX.npy", encoding="bytes" )
        lab = numpy.load( "small_dataset_trainY.npy", encoding="bytes" )
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

class BasicConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch, eps=0.001)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=0.001)
        self.pool = nn.MaxPool2d(stride=2, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.pool(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d( in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1 )
        self.conv2 = nn.Conv2d( in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 )
        self.conv3 = nn.Conv2d( in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1 )
        self.conv4 = nn.Conv2d( in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1 )
        self.conv5 = nn.Conv2d( in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1 )
        self.conv6 = nn.Conv2d( in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 )
        self.lin1 = nn.Linear(200704, 1024)
        self.lin2 = nn.Linear( 1024, 128 )
        self.lin3 = nn.Linear(128,35)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu( self.conv4( x ) )
        x = F.relu( self.conv5( x ) )
        x = self.conv6(x)
        x = self.pool2(x)

        x = x.view( -1, 64 * 56 * 56)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.softmax(self.lin3(x))

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
    avgLoss = 0.0
    cnt = 0
    for i, d in enumerate( train_loader ):
        v = Variable(torch.from_numpy(d[0][0]).float())
        a = Variable(torch.from_numpy(d[0][1]).float())
        # print(v.size())
        # print(a.size())
        lab = d[1]
        y = labelDict[lab]
        y = Variable(torch.from_numpy(numpy.asarray(y).reshape((1))))

        v = v.unsqueeze(0)

        if torch.cuda.is_available():
            net = net.cuda()
            v = v.cuda()
            y = y.cuda()

        out = net(v)
        optimizer.zero_grad()
        loss = criterion( out, y )
        print(loss.data)

        avgLoss += loss.data
        cnt += 1

        loss.backward()
        optimizer.step()

    print("avgLoss")
    print(avgLoss/cnt)
    torch.save(net.state_dict(), str(e)+"image.pt")


