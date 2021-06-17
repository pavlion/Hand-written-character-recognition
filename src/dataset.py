import os

import PIL
import pandas as pd
import os.path as osp

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

filenameToPILImage = lambda x: PIL.Image.open(x).convert('RGB')
# transforms.Normalize(
#    mean=[0.485, 0.456, 0.406], 
#    std=[0.229, 0.224, 0.225])


class FTDataset(Dataset):
    def __init__(self, root_dir, test_ratio=0.2, mode='test', esun_only=False,
        gray_scale=True, train_isNull=True, pretrain=False):

        self.mode = mode
        self.test_ratio = test_ratio
        self.train_isNull = train_isNull # include the 801-th class (is Null)
        self.pretrain = pretrain   # pre-train with 4839 classes
        self.esun_only = esun_only # if esun only, we exclude the open dataset


        self.root_dir = root_dir
        self.data_list, self.index_to_class, self.class_to_index, \
            self.data_stats = self._parse_data(root_dir, test_ratio, mode)
        self.num_class = len(self.data_stats.keys())

        if gray_scale:
            num_channel = 1
            mean_std_set = {"mean": [0.5], "std": [0.5]}
        else:
            num_channel = 3
            mean_std_set = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        
        
        if mode != "test":
            self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.ToTensor(),
                transforms.Resize((50, 50)),
                transforms.Grayscale(num_channel),
                # ThresholdTransform(127), 
                transforms.RandomHorizontalFlip(p=0.35),
                transforms.RandomVerticalFlip(p=0.35),
                #transforms.RandomAffine(translate=(0.05, 0.05)),
                transforms.RandomRotation(degrees=10, 
                    interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize(**mean_std_set)
            ])
        
        else:
            self.transform = transforms.Compose([
                filenameToPILImage,
                transforms.ToTensor(),
                transforms.Resize((50, 50)),
                transforms.Grayscale(num_channel),
                # ThresholdTransform(127), 
                transforms.Normalize(**mean_std_set)
            ])

        
    def __getitem__(self, index):

        file_path, label = self.data_list[index]
        image = self.transform(file_path)
        instance = {'img': image, 'label': label, 'file_path': file_path}
                    
        return instance 

    def __len__(self):
        return len(self.data_list)


    def _parse_data(self, root_dir, test_ratio, mode):
        
        class_list = sorted(os.listdir(root_dir))

        if self.pretrain: # Train for 4839-class classification 

            class_to_index, index_to_class, data_stats = {}, {}, {}
            for c in class_list:
                class_id = int(c.strip().replace("_cropped", "")[1:]) - 1
                index_to_class[class_id] = c
                class_to_index[c] = class_id

        else: # Train for target problem

            class_to_index, index_to_class, data_stats = {}, {}, {}
            for c in class_list:

                id = int(c.strip().replace("_cropped", "")[1:]) - 1
                if self.train_isNull:
                    # words not in the 800 designated words are counted as in the "is Null" class                   
                    index = 0 if id < 800 else 1
                else: 
                    # words not in the designated 800 words are skipped
                    if id >= 800:
                        continue

                    index = id 

                class_to_index[c] = index
                index_to_class[index] = c


        data_list = []
        data_stats = {i: 0 for i in class_to_index.values()}
        for class_name in class_to_index.keys():

            class_id = class_to_index[class_name]
            img_dir = os.path.join(root_dir, class_name)

            # Train-test split
            img_list = os.listdir(img_dir)
            num_img = int(len(img_list) * (1-test_ratio))
            if mode != 'test':
                img_list = img_list[:num_img]
            else:
                img_list = img_list[num_img:]

            data_count = 0
            for file_name in img_list:     

                if self.esun_only and file_name[0] == "c": # it's from open dataset
                    continue

                file_path = os.path.join(img_dir, file_name)
                data_list.append([file_path, class_id]) # file_path, label
                data_count += 1
            

            data_stats[class_id] += data_count
        
        return data_list, index_to_class, class_to_index, data_stats


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    dset = FTDataset(root_dir="croppedData", mode="test", 
        gray_scale=True, train_isNull=True, pretrain=False)
    print(dset.data_stats)

    dset = FTDataset(root_dir="croppedData", mode="test", 
        gray_scale=True, train_isNull=False, pretrain=False)
    print(dset.data_stats)
