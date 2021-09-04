import torch
import torch.nn as nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.sigmoid(self.pool(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pool(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x
