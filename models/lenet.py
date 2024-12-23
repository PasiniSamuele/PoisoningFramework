import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,6,5) #c1:featuremaps 6@28x28 #output = (input-filter)/stride + 1, #filter:5size
        self.conv2 = nn.Conv2d(6,16,5) #c3:feature_maps 16@10x10
        self.maxPool = nn.MaxPool2d(2,2) #subsampling 1/2size
        self.fc1 = nn.Linear(16*5*5,120) #f5:layer120
        self.fc2 = nn.Linear(120,84) #f6:layer84
        self.fc3 = nn.Linear(84,num_classes)  #output:10 class

    def forward(self,x):
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*5*5) #flattens #tensor 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x