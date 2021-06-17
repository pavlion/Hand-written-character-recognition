import os 
import time
import torch
import random
import numpy as np

if os.environ["WANDB"] != '0':
    import wandb
else:
    wandb = None


class Timer:

    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def epoch_time(self):
        curr_time = time.time()
        duration = int(curr_time - self.last_time)
        self.last_time = time.time()

        if duration >= 3600:
            return '{:.1f}h'.format(duration / 3600)
        if duration >= 60:
            return '{}m'.format(round(duration / 60))
        return '{:.2f}s'.format(duration)

    def measure(self, p=1):
        x = (time.time() - self.start_time) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))

        return '{:.2f}s'.format(x)


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed):
    #print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_acc(pred, label, return_raw=False):
    '''
    Calculate the accuracy
    Args:
        pred  (tensor): shape=(batch_size, num_class)
        label (tensor): shape=(batch_size)
    '''
    pred_idx = pred.argmax(dim=1) #
    correct = (pred.argmax(dim=1) == label).int().sum().item()
    total = len(label)

    if return_raw: 
        return correct/total, correct, total
        
    return correct/total

