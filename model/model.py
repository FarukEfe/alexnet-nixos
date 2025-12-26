'''
Architecture
--------
Overlapping pooling: stride < kernel_size achieved by setting s=2 and z=3

Local response normalization: used for generalization, mimics lateral inhibition in real neurons.
                              this is esp. useful for generalization of features. Take squared sum of all activations with
                              kernel i applied at (x,y). Hyperparameters: k=2, n=5, alpha=10e-4, beta=0.75

Double GPU training: convolutional layers are split across 2 GPUs, fully connected layers are repliated on each one.
                     from 2nd conv layer to 3rd, there is cross-GPU communication to share feature maps.

Architecture:

1. Input 3x224x224
2. Conv1: 96 filters (split across 2 GPUs, each with 48 filters), 11x11 kernel, stride 4, padding 0 -> ReLU -> LRN -> MaxPool 3x3, stride 2
3. Conv2: 128 filters each GPU (total 256), 5x5 kernel, stride 1, padding 2 -> ReLU -> LRN -> MaxPool 3x3, stride 2
4. Conv3: 192 filters each GPU (total 384), 3x3 kernel, stride 1, padding 1 -> ReLU
5. Conv4: same as Conv3
6. Conv5: 128 filters each GPU (total 256), 3x3 kernel, stride 1, padding 1 -> ReLU -> MaxPool 3x3, stride 2
7. Dense1: 4096 neurons -> ReLU -> Dropout (p=0.5)
8. Dense2: 4096 neurons -> ReLU -> Dropout (p=0.5)
9. Dense3: 1000 neurons -> Softmax
'''

import torch.nn as nn
import torch.cuda as cuda
import torch as torch
import math, numpy as np

device0 = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() >=2 else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() >=2 else 'cpu')

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pooling=False, lrn=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        if pooling:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        if lrn:
            self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x);
        if hasattr(self, 'lrn'):
            x = self.lrn(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        return x
        
class ConvNet(nn.Module):

    def __init__(self, input_size=(3, 224, 224)):

        super(ConvNet, self).__init__()
        
        if cuda.is_available() and cuda.device_count() >= 2:
            # CNN Split to 2 GPUs
            self.conv1 = ConvLayer(3, 96, kernel_size=11, stride=4, padding=2, pooling=True, lrn=True).to(device0)
            self.conv2d1 = ConvLayer(48, 128, kernel_size=5, stride=1, padding=2, pooling=True, lrn=True).to(device0)
            self.conv2d2 = ConvLayer(48, 128, kernel_size=5, stride=1, padding=2, pooling=True, lrn=True).to(device1)
            self.conv3d1 = ConvLayer(256, 192, kernel_size=3, stride=1, padding=1).to(device0)
            self.conv3d2 = ConvLayer(256, 192, kernel_size=3, stride=1, padding=1).to(device1)
            self.conv4d1 = ConvLayer(192, 192, kernel_size=3, stride=1, padding=1).to(device0)
            self.conv4d2 = ConvLayer(192, 192, kernel_size=3, stride=1, padding=1).to(device1)
            self.conv5d1 = ConvLayer(192, 128, kernel_size=3, stride=1, padding=1, pooling=True).to(device0)
            self.conv5d2 = ConvLayer(192, 128, kernel_size=3, stride=1, padding=1, pooling=True).to(device1)
        else:
            # CPU Architecture
            self.conv1 = ConvLayer(3, 96, kernel_size=11, stride=4, padding=2, pooling=True, lrn=True)
            self.conv2 = ConvLayer(96, 256, kernel_size=5, stride=1, padding=2, pooling=True, lrn=True)
            self.conv3 = ConvLayer(256, 384, kernel_size=3, stride=1, padding=1)
            self.conv4 = ConvLayer(384, 384, kernel_size=3, stride=1, padding=1)
            self.conv5 = ConvLayer(384, 256, kernel_size=3, stride=1, padding=1, pooling=True)

    def forward(self, x):
        if hasattr(self, 'conv2d1') and hasattr(self, 'conv2d2'):
            # Forward pass for 2 GPU architecture
            # Conv1
            x = x.to(device0)
            x = self.conv1(x)
            x0, x1 = torch.split(x, 48, dim=1)  # Split feature maps for 2 GPUs
            # Conv2
            x1 = x1.to(device1)
            x0, x1 = self.conv2d1(x0), self.conv2d2(x1)
            # Conv3 (cross-GPU catenation)
            x0, x1 = torch.cat([x0,x1.to(device0)], dim=1), torch.cat([x0.to(device1),x1], dim=1)
            x0, x1 = self.conv3d1(x0), self.conv3d2(x1)
            # Conv4
            x0, x1 = self.conv4d1(x0), self.conv4d2(x1)
            # Conv5
            x0, x1 = self.conv5d1(x0), self.conv5d2(x1)

            return x0, x1
        else:
            # Forward pass for CPU architecture
            convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
            for conv in convs:
                x = conv(x)
        return x

class FFN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(FFN, self).__init__()
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            # Dense 1
            self.fc1d1 = nn.Linear(input_size, hidden_size//2).to(device0)
            self.fc1d2 = nn.Linear(input_size, hidden_size//2).to(device1)
            # ReLU and Droupout 1
            self.relu1d1 = nn.ReLU().to(device0)
            self.relu1d2 = nn.ReLU().to(device1)
            self.dropout1d1 = nn.Dropout(p=dropout_prob).to(device0)
            self.dropout1d2 = nn.Dropout(p=dropout_prob).to(device1)
            # Dense 2
            self.fc2d1 = nn.Linear(hidden_size, hidden_size//2).to(device0)
            self.fc2d2 = nn.Linear(hidden_size, hidden_size//2).to(device1)
            # ReLU and Dropout 2
            self.relu2d1 = nn.ReLU().to(device0)
            self.relu2d2 = nn.ReLU().to(device1)
            self.dropout2d1 = nn.Dropout(p=dropout_prob).to(device0)
            self.dropout2d2 = nn.Dropout(p=dropout_prob).to(device1)
            # Dense 3 and Softmax (on device0)
            self.fc3 = nn.Linear(hidden_size, output_size).to(device0)
            self.softmax = nn.LogSoftmax(dim=1).to(device0)
        else:
            # Dense 1
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout_prob)
            # Dense 2
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout_prob)
            # Dense 3 and Softmax
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if hasattr(self, 'fc1d1') and hasattr(self, 'fc1d2'):
            # Forward pass for 2 GPU architecture
            x0, x1 = x
            # Dense 1
            x0, x1 = torch.cat([x0, x1.to(device0)], dim=1), torch.cat([x0.to(device1), x1], dim=1)
            x0, x1 = self.fc1d1(x0), self.fc1d2(x1)
            # ReLU and Dropout 1
            x0, x1 = self.relu1d1(x0), self.relu1d2(x1)
            x0, x1 = self.dropout1d1(x0), self.dropout1d2(x1)
            # Dense 2
            x0, x1 = torch.cat([x0, x1.to(device0)], dim=1), torch.cat([x0.to(device1), x1], dim=1)
            x0, x1 = self.fc2d1(x0), self.fc2d2(x1)
            # ReLU and Dropout 2
            x0, x1 = self.relu2d1(x0), self.relu2d2(x1)
            x0, x1 = self.dropout2d1(x0), self.dropout2d2(x1)
            # Dense 3 and Softmax (on device0)
            x0 = torch.cat([x0, x1.to(device0)], dim=1)
            x0 = self.fc3(x0)
            x0 = self.softmax(x0)
            return x0
        else:
            # Dense 1
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            # Dense 2
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            # Dense 3 and Softmax
            x = self.fc3(x)
            x = self.softmax(x)
            return x


class Model(nn.Module):

    def __init__(self, labels=1000):
        super(Model, self).__init__()
        self.convnet = ConvNet()
        self.ffn = FFN(input_size=256*6*6, hidden_size=4096, output_size=labels)

    def forward(self, x):
        x = self.convnet(x)

        # Flatten for FFN
        if isinstance(x, tuple):
            x0, x1 = x
            x0 = x0.view(x0.size(0), -1)
            x1 = x1.view(x1.size(0), -1)
            x = (x0, x1)
        else:
            x = x.view(x.size(0), -1)
            # print(x.shape)
            # exit(1)
        
        x = self.ffn(x)
        return x