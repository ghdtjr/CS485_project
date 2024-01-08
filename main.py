import os
import torch
import argparse
import random
from tqdm import tqdm
import numpy as np

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import *
from utils import *
import matplotlib.pyplot as plt

def plot_save(x, y, xlabel, ylabel, title=''):
    x = [str(element) for element in x]
    plt.plot(x, y)
    if title == '':
        plt.title(f'{xlabel} vs {ylabel}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(f'./results/{xlabel + ylabel}.png')
        plt.show()
    else:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(f'./results/{xlabel + ylabel}.png')
        plt.show()
  
def get_mean_std(dataset):
    meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]
    stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]
    
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    
    means = [meanR, meanG, meanB]
    stds = [meanR, meanG, meanB]
    return means, stds
    
     
def load_data(width=224, height=224, batch_size=16, num_workers=1):
    
    #########################
    # normalization
    #########################
    transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor()
    ])
    
    train_imgfolder = ImageFolder(os.path.join(f'./train_data'), transform)
    test_imgfolder = ImageFolder(os.path.join(f'./test_data'), transform)
    
    means, stds = get_mean_std(train_imgfolder)
    
    transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    
    #########################
    # Reload data with normalization
    #########################
    train_imgfolder = ImageFolder(os.path.join(f'./train_data'), transform)
    test_imgfolder = ImageFolder(os.path.join(f'./test_data'), transform)
    
    train_dataloader = DataLoader(
        dataset=train_imgfolder,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_imgfolder,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_dataloader, test_dataloader
    
def train(model, optimizer, args, train_loader, device):
    if args.loss_fun == 'entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_fun == 'hinge':
        criterion = nn.MultiMarginLoss(p=2)
    else:
        raise NotImplementedError
        # criterion
    
    model.train()
    for epoch in tqdm(range(args.num_epochs)):
        print('epoch:', epoch)
        
        total_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('raw outputs:', outputs)
            # print('raw outputs:', outputs.shape)
            # _, predictions = torch.max(outputs, 1)
            # print('predictions:', predictions)
            # print('predictions:', predictions.shape)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(total_loss)
    
    torch.save(model.state_dict(), 'model1.pth')
            
            
def test(model, test_loader, device):
    print('start test')
    model.eval()
    with torch.no_grad():
        total_acc = 0.0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            print('target', targets.shape)
            outputs = model(data, mode='compression')
            print('outputs', outputs.shape)
            _, predictions = torch.max(outputs, 1)
            batch_acc = accuracy(predictions, targets)
            total_acc += batch_acc * targets.shape[0]
            
        total_acc /= 150
        
    print('accuracy:', total_acc)
    print('\n\n')
    with open(os.path.join('logs.txt'), 'a') as f:
        log_data = f"accuracy is {total_acc}"
        f.write(log_data)
    return total_acc
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # train options setting
    parser.add_argument('--gpu_num', type=int, default=1)
    
    parser.add_argument('--num_epochs', default=99, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--compression', default=0.3, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--kernel_size', default=3, type=int)
    
    parser.add_argument('--pre_train', default='CNN scratch', type=str)
    
    # loss function
    parser.add_argument('--loss_fun', default='entropy', type=str)
    
    parser.add_argument('--norm', default='batch norm', type=str)
    
    parser.add_argument('--target', default='', type=str)
    args = parser.parse_args()
    
    log_args_time(args)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device

    train_loader, test_loader = load_data(batch_size = args.batch_size)

    
    drop_outs = [x / 100.0 for x in range(0, 99, 10)]
    num_epochs = range(10, 200, 10)
    lrs = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    weight_decays = [x / 100.0 for x in range(0, 99, 10)]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    loss_functions = ['entropy', 'hinge']
    norms = ['batch norm', 'group norm']
    inits = ['CNN scratch', 'pretrained alex']
    # kernels = [3, 6, 9, 12, 15]
    kernels = [3, 6, 9, 12, 15]
    kernels = [3]
    preserve = [x / 100.0 for x in range(10, 99, 10)]
    
    if args.target == 'Drop_out':
        target_list = drop_outs
    if args.target == 'Num_Epoch':
        target_list = num_epochs
    if args.target == 'Learning_Rate':
        target_list = lrs
    if args.target == 'Regularization':
        target_list = weight_decays
    if args.target == 'Batch_Size':
        target_list = batch_sizes
    if args.target == 'Loss_Function':
        target_list = loss_functions
    if args.target == 'Nomalization':
        target_list = norms
    if args.target == 'Initialization':
        target_list = inits
    if args.target == 'Kernel_Size':
        target_list = kernels
    if args.target == 'Compression':
        target_list = preserve
    else:
        target_list = [1]
        
    results = []
    for i, value in enumerate(target_list):
        if args.target == 'Drop_out':
            args.dropout = value
        if args.target == 'Num_Epoch':
            args.num_epochs = value
        if args.target == 'Learning_Rate':
            args.lr = value
        if args.target == 'Regularization':
            args.weight_decay = value
        if args.target == 'Batch_Size':
            args.batch_size = value
        if args.target == 'Loss_function':
            args.loss_fun = value
        if args.target == 'Nomalization':
            args.norm = value
        if args.target == 'Initialization':
            args.pre_train = value
        if args.target == 'Kernel_Size':
            args.kernel_size = value
        if args.target == 'Compression':
            args.compression = value
        else:
            print('else')

        model = CNN(args)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        #data_loader
        train(model, optimizer, args, train_loader, device)
        acc = test(model, test_loader, device).cpu()
        results.append(acc)
        
    plot_save(target_list, results, 'Compression Ratio', 'Accuracy')
    
