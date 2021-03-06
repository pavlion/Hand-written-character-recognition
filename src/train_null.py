import os 
import torch
import torch.nn as nn
from tqdm import tqdm
from adabelief_pytorch import AdaBelief

from dataset import FTDataset
from model import FTModel, num_param
from loss import Loss
from utils import set_random_seed, _ensure_dir, Timer
from logger import TrainingLogger, MetricMeter
from stopper import EarlyStopping



def train(config):
    
    _ensure_dir(config.dest_path)
    logger = TrainingLogger(os.path.join(config.dest_path, "log.txt"))
    
    logger.print("Set random seed to {}".format(config.seed))
    set_random_seed(config.seed)

    train_dset = FTDataset(
        root_dir=config.root_dir, 
        test_ratio=config.test_ratio, 
        gray_scale=config.gray_scale,
        include_null_class=True, 
        mode='train')
    val_dset = FTDataset(
        root_dir=config.root_dir, 
        test_ratio=config.test_ratio,
        gray_scale=config.gray_scale,
        include_null_class=True, 
        mode='test')
    num_class = train_dset.num_class 

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dset,
        batch_size=config.test_batch_size,
        shuffle=False
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = FTModel(num_class=num_class, model_type=config.model_type, pretrain=config.pretrain)
    
    criterion = Loss(loss_type=config.loss_type)
    if config.optim_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim_type == "AdaBelief":
        optimizer = AdaBelief(model.parameters(), lr=config.lr, print_change_log=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    if config.data_parallel:
        model = nn.DataParallel(model)
    
    model.device = device


    logger.print(f"Number of classes: {num_class}")
    logger.print(f"Number of model parameters: {num_param(model)}")
    logger.print('Length of train/val datasets: {}, {}'.format(
        len(train_dset), len(val_dset)))
    logger.print('Length of train/val dataloaders: {}, {}'.format(
        len(train_loader), len(val_loader)))
    logger.print(f'Using {torch.cuda.device_count()} GPUs: '
        + f'{", ".join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])}')
    
    f = open(os.path.join(config.dest_path, "model_arch.txt"), "w")
    print(model, file=f)
    f.close()
    
    
    train_class_list = train_dset.data_stats.values()
    test_class_list = val_dset.data_stats.values()
    logger.print("Max/min/avg number of train datasets: {}, {}, {}".format(
        max(train_class_list), min(train_class_list), sum(train_class_list)/len(train_class_list)
    ))
    logger.print("Max/min/avg number of test datasets: {}, {}, {}".format(
        max(test_class_list), min(test_class_list), sum(test_class_list)/len(test_class_list)
    ))

    logger.print('Config:\n' + str(config))
    logger.print('\n')


    timer = Timer()
    min_loss, best_acc = 1e10, 0.0
    early_stopper = EarlyStopping(patience=50, delta=0, print_fn=logger.print)
    model = model.to(device)

    for epoch in range(1, config.num_epoch+1):
        
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        
        ####### Train #######
        meters = {
            "loss": MetricMeter(),
            "acc": MetricMeter(),
            "f1": MetricMeter()
        }
        train_steps = 0
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader, start=1):
            img = batch['img'].to(device)
            label = batch['label'].to(device)
            logits = model(img)

            loss = criterion(logits, label)/config.grad_accum_step
            loss.backward()

            meters["loss"].update(
                correct=loss.item(), 
                total=1)
            meters["acc"].update(
                correct=(logits.argmax(dim=1) == label).int().sum().item(), 
                total=len(label))
            # print((logits.argmax(dim=1) == label).int().sum().item(), len(label))
            
            if i % config.grad_accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                train_steps += 1
                
                logger.print('Epoch {:3d} | Train step {:3d} | Loss: {:5f} | Acc: {:5f}%'.format(epoch, 
                    train_steps, meters["loss"].get_score(), meters["acc"].get_score()*100.0), end="\r")

        train_loss, train_acc = meters["loss"].get_score(), meters["acc"].get_score()
        logger.print('Epoch {:3d} | Train | Loss: {:5f} | Acc: {:5f}%'.format(
            epoch, train_loss, train_acc*100.0))


        ####### Validation #######
        meters = {
            "loss": MetricMeter(),
            "acc": MetricMeter(),
            "f1": MetricMeter()
        }
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader, start=1):
                img = batch['img'].to(device)
                label = batch['label'].to(device)
                logits = model(img)
                loss = criterion(logits, label)

                meters["loss"].update(
                    correct=loss.item(), 
                    total=1)
                meters["acc"].update(
                    correct=(logits.argmax(dim=1) == label).int().sum().item(), 
                    total=len(label))

            

            val_loss, val_acc = meters["loss"].get_score(), meters["acc"].get_score() 
            logger.print('Epoch {:3d} |  Val  | Loss: {:5f} | Acc: {:5f}%'.format(
                epoch, val_loss, val_acc*100.0))  

            if val_loss < min_loss:
                min_loss = val_loss
                best_acc = val_acc
                save_checkpoint(model, os.path.join(config.dest_path, "best_loss.pth"))
                save_checkpoint(model, os.path.join(config.dest_path, "best_acc.pth"))
                logger.print("Model(loss) saved.")

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, os.path.join(config.dest_path, "best_acc.pth"))
                logger.print("Model(acc) saved.")

        early_stopper(val_loss, val_acc)
        if early_stopper.early_stop:
            break
        
        logger.print('Current best: min_loss = {:.5f}, best_acc = {:.5f}%'.format(
            min_loss, best_acc*100.0))
        logger.print("Epoch time: {}/{}".format(timer.measure(p=1), 
            timer.measure(p=epoch/config.num_epoch)))
        logger.print("\n")

        

    return min_loss, best_acc


def save_checkpoint(model, path):

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path):

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))