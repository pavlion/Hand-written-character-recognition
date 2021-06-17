import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Loss(torch.nn.Module):
    def __init__(self, loss_type="CE", **kwargs):
        super().__init__()

        if loss_type == "CE":
            self.loss = torch.nn.CrossEntropyLoss(**kwargs)
        elif loss_type == "F1":
            self.loss = F1_Loss()
    
    def forward(self, pred, target, **kwargs):
        return self.loss(pred, target, **kwargs)
    


def f1_loss(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=True) -> torch.Tensor:
    '''Calculate F1 score. Works with gpu tensors
    Return f1 (torch.Tensor): f1 score, `ndim` == 1, 0 <= val <= 1
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    print(precision, recall)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    return f1


class F1_Loss(torch.nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        num_class = y_pred.size(1)
        y_true = F.one_hot(y_true, num_class).to(torch.float32)
        # print(y_true.shape)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
