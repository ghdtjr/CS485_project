import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def truncated_svd(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    U, s, V = torch.svd(W, some=True)

    Ul = U[:, :l]
    sl = s[:l]
    V = V.t()
    Vl = V[:l, :]

    SV = torch.mm(torch.diag(sl), Vl)
    return Ul, SV


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
                
                # input_size = 100352
                flat_size = 31 - kernel_size
                fc1_size = 128*flat_size*flat_size

                self.fc1 = nn.Linear(fc1_size, 1024)
                self.drop = nn.Dropout(args.dropout)
                self.fc2 = nn.Linear(1024, 10)
                
            elif args.norm == 'group norm':
                num_groups = 4
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding=1)
                self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=32)  # GroupNorm layer
                
                # Max Pooling Layer
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # Convolutional Layer 2
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1)
                self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=64)  # GroupNorm layer
                
                # Convolutional Layer 3
                self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1)
                self.gn3 = nn.GroupNorm(num_groups=num_groups, num_channels=128)  # GroupNorm layer
                
                self.fc1 = nn.Linear(100352, 1024)
                
                
                self.drop = nn.Dropout(args.dropout)
                self.fc2 = nn.Linear(1024, 10)
                
        elif args.pre_train == 'pretrained alex':
            self.model = models.alexnet(pretrained = True)
            self.model.classifier[6] = nn.Linear(4096, 10)
            
        
    def forward(self, x, mode='no'):
        if self.args.pre_train == 'CNN scratch':
            if self.args.norm == 'batch norm':
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                print(x.shape)
                flat_size = 31 - self.kernel_size
                if self.kernel_size > 9:
                    flat_size = 32 - self.kernel_size
                x = x.view(-1, 128 * flat_size * flat_size)  # Adjust the flattening size accordingly
                
                if mode == 'compression':
                    preserve_ratio = 1 - self.args.compression
                    gemm_weights = self.fc1.weight
                    self.U, self.SV = truncated_svd(gemm_weights.data, int(preserve_ratio * gemm_weights.size(0)))
                    
                    self.fc_u = nn.Linear(self.U.size(1), self.U.size(0)).to(self.args.device)
                    self.fc_u.weight.data = self.U
            
                    self.fc_sv = nn.Linear(self.SV.size(1), self.SV.size(0)).to(self.args.device)
                    self.fc_sv.weight.data = self.SV
                    x = self.fc_sv.forward(x)
                    x = self.fc_u.forward(x)
                    print('here?')
                else:
                    x = F.relu(self.fc1(x))
                x = self.drop(x)
                x = self.fc2(x)
                return x
            elif self.args.norm == 'group norm':
                x = self.pool(F.relu(self.gn1(self.conv1(x))))
                x = self.pool(F.relu(self.gn2(self.conv2(x))))
                x = self.pool(F.relu(self.gn3(self.conv3(x))))
            
                x = x.view(-1, 128 * 28 * 28)  # Adjust the flattening size accordingly
                x = F.relu(self.fc1(x))
                x = self.drop(x)
                x = self.fc2(x)
                return x
            
        elif self.args.pre_train == 'pretrained alex':
            return self.model(x)
