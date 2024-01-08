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
                
                # Convolutional Layer 2
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1)
                self.bn2 = nn.BatchNorm2d(num_features=64)
                
                # Convolutional Layer 3
                self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1)
                self.bn3 = nn.BatchNorm2d(num_features=128)

                # Convolutional Layer 4
                self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=1)
                self.bn4 = nn.BatchNorm2d(num_features=256)

                # Convolutional Layer 5
                self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=1)
                self.bn5 = nn.BatchNorm2d(num_features=512)

                # Convolutional Layer 6
                self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=kernel_size, padding=1)
                self.bn6 = nn.BatchNorm2d(num_features=1024)

                # Convolutional Layer 7
                self.conv7 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=kernel_size, padding=1)
                self.bn7 = nn.BatchNorm2d(num_features=2048)

                # Max Pooling Layer
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # The input size here is an example, adjust based on your input dimensions
                input_size = 2048 * 1 * 1  # Adjust this based on your input image size and network architecture
                self.fc1 = nn.Linear(input_size, 1024)
                self.drop = nn.Dropout(args.dropout)
                self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        if self.args.pre_train == 'CNN scratch':
            if self.args.norm == 'batch norm':
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                x = self.pool(F.relu(self.bn4(self.conv4(x))))
                x = self.pool(F.relu(self.bn5(self.conv5(x))))
                x = self.pool(F.relu(self.bn6(self.conv6(x))))
                x = self.pool(F.relu(self.bn7(self.conv7(x))))
                
                print(x.shape)
                # Adjust the flattening size accordingly
                x = x.view(-1, 2048 * 1 * 1) 
                x = F.relu(self.fc1(x))
                x = self.drop(x)
                x = self.fc2(x)
                return x
