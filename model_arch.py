"""
All model architecture definitions
"""

import torch
import torch.nn as nn


class DQN_CNN(nn.Module):
    def __init__(self,num_actions):
        # input to the network is 84 x 84 x 4
        super().__init__()
        self.conv1 = nn.Conv2d(4,32,(8,8),stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32,64,(4,4),stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64,64,(3,3),stride=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64*7*7,512)
        self.fc2 = nn.Linear(512,num_actions)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        #print('x.shape=',x.shape)
        x = torch.flatten(x,1)
        #print('x.shape=',x.shape)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
