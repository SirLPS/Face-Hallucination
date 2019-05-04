# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:26:33 2019

@author: lyfeng
"""

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data


class CASIAWebFace_dataset(data.Dataset):
    def __init__(self,root,file_list,transform=None,downsample=False):
        self.root = root
        self.transform = transform        
        self.downsample = downsample
        self.image_list = []
        self.label_list = []
        
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        
        for info in img_label_list:
            info = info.split('\t')
            image_path, label_name = info[0], info[1]           
            self.image_list.append(image_path)
            self.label_list.append(int(label_name))
            
        self.class_nums = len(np.unique(self.label_list))
        print("Totally {:d} images of {:d} identites in the dataset."\
              .format(len(self.image_list), self.class_nums))
        
    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        img = Image.open(self.root+'/'+img_path).convert('RGB')
        if self.downsample:
            img_in = img.resize((12,14), resample=Image.BICUBIC)
#        img = (img-127.5)/128.0
#        if img.shape[0]==1:          # gray images!
#            img = np.stack([img]*3,2)
        
        flip = np.random.random()>0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)   # flip horizontally
            if self.downsample:
                img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform:
            img = self.transform(img)
            if self.downsample:
                img_in = self.transform(img_in)
        if self.downsample:
            return img_in, img, torch.Tensor([label])
        return img, torch.Tensor([label])
    
    def __len__(self):
        return len(self.image_list)       
            
    

   
