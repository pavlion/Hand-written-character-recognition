import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB'] = '0'

from train import train
from config import Config

config = Config()

config.seed = 2021
config.root_dir = "croppedData"
config.train_batch_size = 10000
config.test_batch_size = 10000
config.lr = 1e-3
config.test_ratio = 0.2
config.num_epoch = 200
config.grad_accum_step = 1
config.data_parallel = False
config.optim_type = "Adam"
config.loss_type = "CE"
config.model_type = "resnet18"
config.gray_scale = False
config.pretrain = True
config_cfg_str = ""
config_cfg_str += "gray" if config.gray_scale else ""
config_cfg_str += "_pretrain" if config.pretrain else ""

current_time = time.strftime("%m%d_%H%M") #%Y%S
current_time = "0"
config.dest_path = os.path.join("ckpt", current_time)


f = open("performance.csv", "a")

performance = [] 
for model_type in ["resnet18"]: 
    for lr in [1e-3]:

        config.model_type = model_type
        config.lr = lr
        config.dest_path = os.path.join("ckpt", f"{model_type}_lr{lr}_{config.loss_type}_{config_cfg_str}")

        min_loss, best_acc = train(config)

        config.dump(os.path.join(config.dest_path, "config.json"))
        print("{}, lr={}, loss_type={}, min_loss={:6f}, best_acc={:6f}%\n".format(
            model_type, lr, config.loss_type, min_loss, best_acc*100))

f.close()

    
