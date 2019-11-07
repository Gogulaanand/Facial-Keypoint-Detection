## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
         
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints       
        
        self.conv1 = nn.Conv2d(1,32,5)       # (224-5)/1+1 = (32,92,92)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)      # floor(220/2) => (32,46,46)
        self.conv2 = nn.Conv2d(32,64,5)        # (110-5) +1 => (64,88,88) => (64,44,44)
        self.conv3 = nn.Conv2d(64,64,5)        # (53-5)+1 => (64,22,17) => (64,11,8)
        self.fc1 = nn.Linear(64*8*8,512)
        self.fc2 = nn.Linear(512,136)
        self.dp = nn.Dropout(p=0.2)
           

        
    def forward(self, x):
        #feedforward behavior of this model
        ## x is the input image 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dp(self.pool(F.relu(self.conv2(x))))
        x = self.dp(self.pool(F.relu(self.conv3(x))))
        x = x.view(x.size(0),-1)
        x = self.dp(self.fc1(x))
        x = self.fc2(x)        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
