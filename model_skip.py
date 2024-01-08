import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        
        self.args = args
        kernel_size = args.kernel_size
        self.kernel_size = kernel_size
        if args.pre_train == 'CNN scratch':
            if args.norm == 'batch norm':
                # Convolutional Layer 1
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding=1)
                self.bn1 = nn.BatchNorm2d(num_features=32)
                # Max Pooling Layer
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # Convolutional Layer 2
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1)
                self.bn2 = nn.BatchNorm2d(num_features=64)
                
                # Convolutional Layer 3
                self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1)
                self.bn3 = nn.BatchNorm2d(num_features=128)
                
                self.adjust_conv = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1)

                # input_size = 100352
                flat_size = 31 - kernel_size
                fc1_size = 128*flat_size*flat_size

                self.fc1 = nn.Linear(fc1_size, 1024)
                self.drop = nn.Dropout(args.dropout)
                self.fc2 = nn.Linear(1024, 10)
                
            
        
    def forward(self, x):
        if self.args.pre_train == 'CNN scratch':
            if self.args.norm == 'batch norm':
                print('first x shape', x.shape)
                x1 = self.pool(F.relu(self.bn1(self.conv1(x))))
                print('first x1 shape', x1.shape)
                x = self.pool(F.relu(self.bn2(self.conv2(x1))))
                print('second x shape', x.shape)
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                
                # Adjust and add the skip connection
                x1_adjusted = self.adjust_conv(x1)
                x1_adjusted = self.pool(x1_adjusted)
                x1_adjusted = self.pool(x1_adjusted)
                print('third x shape', x.shape)
                print(x1_adjusted.shape)
                x = x + x1_adjusted  # Element-wise addition

                print(x.shape)
                flat_size = 31 - self.kernel_size

                x = x.view(-1, 128 * flat_size * flat_size)  # Adjust the flattening size accordingly
                x = F.relu(self.fc1(x))
                x = self.drop(x)
                x = self.fc2(x)
                return x
