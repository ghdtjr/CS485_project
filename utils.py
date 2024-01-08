import torch
import os
from datetime import datetime
from pytz import timezone


def accuracy(predictions, labels):
    train_acc_num = torch.sum(predictions == labels)
    acc = train_acc_num / labels.shape[0]
    return acc

def log_args_time(args):
    with open(('./logs.txt'), 'a') as f:
        f.write("\n--------------------------------------------------------")
        now = datetime.now(timezone('Asia/Seoul'))
        f.write("Timestamp: {}_{}-{}_{} \n".format(now.month, now.day, now.hour, now.minute))
        f.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    f.close()