import os
from numpy import load 
import torch
import torch.nn as nn
from tqdm import tqdm
from adabelief_pytorch import AdaBelief
from sklearn.metrics import f1_score

from dataset import FTDataset
from model import FTModel, num_param
from loss import Loss, f1_loss
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
        train_isNull=config.train_isNull, # train Null/is Null class
        pretrain=config.pretrain, # train using 4839 classes
        esun_only=config.esun_only, # exclude open dataset
        mode='train')
    val_dset = FTDataset(
        root_dir=config.root_dir, 
        test_ratio=config.test_ratio,
        gray_scale=config.gray_scale,
        train_isNull=config.train_isNull, # train Null/is Null class
        pretrain=config.pretrain, # train using 4839 classes
        esun_only=config.esun_only, # exclude open dataset
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
    model = FTModel(num_class=num_class, model_type=config.model_type, 
        pretrain_model_path=config.model_path, imagenet_pretrain=config.use_imagenet_pretrain)
        
    class_weight = [config.trg_class_weight] * 800 + [1.0] * (num_class-800)
    class_weight = torch.FloatTensor(class_weight).to(device)
    criterion = Loss(loss_type=config.loss_type, weight=class_weight)
    if config.optim_type == "AdaBelief":
        optimizer = AdaBelief(model.parameters(), lr=config.lr, print_change_log=False)
    else:
        optim_ptr = getattr(torch.optim, config.optim_type)
        optimizer = optim_ptr(model.parameters(), lr=config.lr)
    
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
    min_loss, best_acc, best_f1, best_new_acc, best_new_f1 = 1e10, 0.0, 0.0, 0.0, 0.0
    early_stopper = EarlyStopping(patience=25, delta=0, print_fn=logger.print)
    model = model.to(device)

    for epoch in range(1, config.num_epoch+1):
        
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        
        ####### Train #######
        meters = {
            "loss": MetricMeter(),
            "acc": MetricMeter(),
            "f1": MetricMeter()
        }
        train_iter(model, train_loader, criterion, optimizer, meters, logger, config)
        train_loss, train_acc = meters["loss"].get_score(), meters["acc"].get_score()
        logger.print('Epoch {:3d} | Train | Loss: {:5f} | Acc: {:5f}%'.format(
            epoch, train_loss, train_acc*100.0))


        ####### Validation #######
        meters = {
            "loss": MetricMeter(),
            "acc": MetricMeter(),
            "f1": MetricMeter()
        }
        val_iter(model, val_loader, criterion, meters)
        val_loss, val_acc, val_f1 = meters["loss"].get_score(), meters["acc"].get_score(), meters["f1"].get_score()
        new_val_acc, new_val_f1 = meters["New acc"], meters["New f1"] 
        logger.print('Epoch {:3d} |  Val  | New Acc: {:5f}% | New F1: {:.5f}%'.format(
            epoch, new_val_acc, new_val_f1*100.0)) 

        if val_loss < min_loss:
            min_loss = val_loss
            save_checkpoint(model, os.path.join(config.dest_path, "best_loss.pth"))
            # save_checkpoint(model, os.path.join(config.dest_path, "best_acc.pth"))
            logger.print("Model(loss) saved.")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, os.path.join(config.dest_path, "best_acc.pth"))
            logger.print("Model(acc) saved.")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, os.path.join(config.dest_path, "best_f1.pth"))
            logger.print("Model(f1) saved.")

        if new_val_acc > best_new_acc:
            best_acc = val_acc
            save_checkpoint(model, os.path.join(config.dest_path, "best_801_acc.pth"))
            logger.print("Model(new acc) saved.")
        
        if new_val_f1 > best_new_f1:
            best_f1 = val_f1
            save_checkpoint(model, os.path.join(config.dest_path, "best_801_f1.pth"))
            logger.print("Model(new f1)) saved.")
        


        early_stopper(val_loss, val_acc)
        logger.print('Current best: min_loss = {:.5f}, best_acc = {:.5f}%, best f1 = {:.5f}%'.format(
            min_loss, best_acc*100.0, best_f1*100.0))
        logger.print("Epoch time: {}/{}".format(timer.measure(p=1), 
            timer.measure(p=epoch/config.num_epoch)))
        
        if early_stopper.early_stop:
            break

        logger.print("\n")



    f1, metadata = val_special(model, val_loader)    
    mean = lambda x: sum(x)/len(x)
    acc = [correct/total*100 for correct, total in metadata]

    logger.print("Overall F1 score: {}%".format(f1*100))

    f = open(os.path.join(config.dest_path, "class_acc.txt"), "w")
    print(acc, file=f)
    f.close()

    return min_loss, best_acc, best_f1


def train_iter(model, train_loader, criterion, optimizer, meters, logger, config):
    device = model.device
    train_steps = 0
    grad_accum_step = min(config.grad_accum_step, len(train_loader))
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader, start=1):
        img = batch['img'].to(device)
        label = batch['label'].to(device)
        logits = model(img)
        
        loss = criterion(logits, label)/grad_accum_step
        loss.backward()

        meters["loss"].update(
            correct=loss.item()*grad_accum_step, 
            total=1
        )
        meters["acc"].update(
            correct=(logits.argmax(dim=1) == label).int().sum().item(), 
            total=len(label)
        )
        # print((logits.argmax(dim=1) == label).int().sum().item(), len(label))
        
        if i % grad_accum_step == 0 or i == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            train_steps += 1
            
            logger.print('Train step {:3d} | Loss: {:5f} | Acc: {:5f}%'.format( 
                train_steps, meters["loss"].get_score(), meters["acc"].get_score()*100.0), end="\r")


def val_iter(model, val_loader, criterion, meters):
    device = model.device
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader, start=1):
            img = batch['img'].to(device)
            label = batch['label'].to(device)
            logits = model(img)
            loss = criterion(logits, label)

            meters["loss"].update(
                correct=loss.item()*len(label), 
                total=len(label)
            )
            meters["acc"].update(
                correct=(logits.argmax(dim=1) == label).int().sum().item(), 
                total=len(label)
            )

            labels.append(label.cpu())
            preds.append(logits.argmax(dim=1).cpu())
        
    labels, preds = torch.cat(labels, dim=0), torch.cat(preds, dim=0)
    f1 = f1_score(labels, preds, average="macro")
    meters["f1"].update(
        correct=f1.item(), 
        total=1
    )

    # for i in range(len(preds)):
    #     p = preds[i].item()              
    #     l = label[i].item()
    #     preds[i] = p if p < 800 else 800
    #     label[i] = l if l < 800 else 800
    preds = torch.clamp(preds, max=800)
    labels = torch.clamp(labels, max=800)
    meters["New acc"] = (preds == labels).int().sum().item()
    meters["New f1"] = f1_score(labels, preds, average="macro")


def val_special(model, val_loader):
    device = model.device
    model.to(device)
    model.eval()
    labels, preds = [], []
    metadata = [[1e-7,1e-7] for _ in range(801)] # correct, total

    with torch.no_grad():
        for i, batch in enumerate(val_loader, start=1):
            img = batch['img'].to(device)
            label = batch['label']
            logits = model(img)
            pred = logits.argmax(dim=1).cpu()

            for l, p in zip(label, pred):
                # 800~4039 belongs to NULL
                l = l if l < 800 else 800 
                p = p if p < 800 else 800

                metadata[l][1] += 1 
                if l == p: metadata[l][0] += 1

            labels.append(label)
            preds.append(pred)
        
        labels, preds = torch.cat(labels, dim=0), torch.cat(preds, dim=0)
        f1 = f1_score(labels, preds, average="macro")
        
        return f1, metadata


def save_checkpoint(model, path):

    print("Save model checkpoint to {}".format(path))
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    
    print("Load model checkpoint from {}".format(path))
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))



if __name__ == "__main__":

    from config import Config
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--config_path", type=str, required=True)

    config = Config()
    config.load(args.config_path)
    train(config)


