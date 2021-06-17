import os
import time
import subprocess 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB'] = '0'

from train import train
from config import Config

config = Config()

### Training settings
config.seed = 2021
config.root_dir = "croppedData"
config.train_batch_size = 10000 # 3000
config.test_batch_size = 10000
config.lr = 1e-3
config.test_ratio = 0.2
config.num_epoch = 250
config.grad_accum_step = 2 # 12
config.data_parallel = False
config.optim_type = "AdamW"
config.loss_type = "F1"

### Problem settings
config.train_isNull = False # train isNull or 800 classes
config.gray_scale = False   # whether to use gray scale img
config.pretrain = True      # train 4039 classes
config.use_imagenet_pretrain = True
config.esun_only = False

### Model settings
config.model_type = "resnet101"
config.model_path = None
config.trg_class_weight = 1.5


# config.cfg_str += "NULL_" if config.train_isNull else "800class_"
# config.cfg_str += "gray_" if config.gray_scale else ""
# config.cfg_str += "use_imagenet_pretrain_" if config.use_imagenet_pretrain else ""






f = open("performance.csv", "a")
model_type = "resnet18"
performance = [] 
for trg_class_weight in [1.2]: # , "resnet50"
    for lr in [1e-3]:
        for loss_type in ["CE"]: # , "F1"
            
            current_time = time.strftime("%m%d_%H%M") #%Y%S
            # current_time = "0"
            config.cfg_str = "train4839_HVflip_" + current_time

            config.loss_type = loss_type
            config.trg_class_weight = trg_class_weight
            config.model_type = model_type
            config.lr = lr
            
            config.dest_path = os.path.join("ckpt", 
                f"{model_type}_lr{lr}_{config.loss_type}_weight{trg_class_weight}_{config.cfg_str}")



            min_loss, best_acc, best_f1 = train(config)

            config.dump(os.path.join(config.dest_path, "config.json"))
            print("{}, lr={}, loss_type={}, min_loss={:6f}, best_acc={:6f}%, best_f1={:6f}%, ckpt_dir={}".format(
                model_type, lr, config.loss_type, 
                min_loss, best_acc*100.0, best_f1*100.0, 
                f"{model_type}_lr{lr}_{config.loss_type}_{config.cfg_str}"), file=f)

f.close()

    
