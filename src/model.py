import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, resnet50, resnet101, resnet152, inception_v3, googlenet
# import models.resnet_gray as resnet_gray
# import models.resnet_deform as resnet_deform 
#from models import resnet18_gray, resnet34_gray, resnet50_gray, resnet101_gray, resnet152_gray


num_param = lambda model: sum([p.numel() for p in model.parameters()])

class FTModel(nn.Module):
    '''
    Args:
        pretrain_model_path: path to the pre-trained model that was trained on 4839 classes
        imagenet_pretrain: whether to load imagenet pre-train models
    '''
    def __init__(self, num_class, model_type="resnet18", 
        pretrain_model_path=None, imagenet_pretrain=False, **kwargs):

        super().__init__()
        self.num_class = num_class
        
        if "deform" in model_type:

            self.model = getattr(resnet_deform, model_type)(pretrained=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_class)

        elif "gray" in model_type:
            
            model_ptr = getattr(resnet_gray, model_type.replace("_gray", ""))
            self.model = model_ptr(pretrained=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_class)
        
        else: # model_type in ["resnet18", "resnet50", "resnet101", "resnet152", "googlenet"]: 
            # "inception_v3" cannot be used: img too small

            self.model = getattr(torchvision.models, model_type)(pretrained=imagenet_pretrain)
            # in_features = self.model.fc.in_features
            # self.model.fc = nn.Linear(in_features, num_class)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain_model_path:

            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 4839)
            self.load_state_dict(torch.load(pretrain_model_path))
            self.model.fc = nn.Linear(in_features, num_class)
        
        else:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_class)


        if False: #pretrain_model_path:
            pretrained_dict = torch.load(pretrain_model_path)
            model_state = self.state_dict()
            old_model_state = self.state_dict()

            for k, v in pretrained_dict.items():
                # k = k[len("model."):]
                if not (k in model_state): 
                    print(f"Key {k} not in model state")
                else:
                    if (model_state[k].shape != pretrained_dict[k].shape):
                        print(f"Param of {k} has different shapes")

            
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_state) and (model_state[k].shape == pretrained_dict[k].shape)}
            model_state.update(pretrained_dict)
            self.load_state_dict(model_state)

            for (k1, v1), (k2, v2) in zip(old_model_state.items(), model_state.items()):
                

                if k1 != k2:
                    print(f"Unequal key: {k1}, {k2}")
                    
                else:
                    if ((v1.cpu() != v2.cpu()).int().sum()) != 0:
                        print(f"Key {k1} is updated!")
                    else:
                        continue
                
                print("-"*30)

                


    def forward(self, img, **kwargs):
        
        logits = self.model(img, **kwargs)

        return logits


if __name__ == '__main__':
    shots = torch.rand((10, 3, 84, 84)).cuda()
    query = torch.rand((15, 3, 84, 84)).cuda()
    
    pretrain_model_path = "ckpt/resnet18_4839_pretrain/best_loss.pth"
    model = FTModel(num_class=2, model_type="resnet18", pretrain_model_path=pretrain_model_path)
    print(model)