# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:37:06 2019

@author: lyfeng
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import os,sys,cv2,random,datetime
import argparse
import CNN_H
from CASIAWebFace import CASIAWebFace_dataset
import torchvision.transforms as transforms
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='Hallu_Net', type=str)
#parser.add_argument('--dataset', default='../../dataset/face/casia/casia.zip', type=str)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=32, type=int, help='')
parser.add_argument('--data_root',default='../datasets/CASIA-WebFace-aligned')
parser.add_argument('--file_root',default='../datasets/casia_landmark.txt')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
])
dataset = CASIAWebFace_dataset(args.data_root, args.file_root, transform=transform, downsample=True)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=False)
 


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()


def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')
    

def train(net,epoch,args,train_loader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    
    pbar = enumerate(train_loader)
    for batch_idx, data in pbar:
        
        img_in,img,label = data     # img_in is downsampled x8 from img
        if img is None: break       
        inputs = img_in.float()
        targets = img.float()
#        targets = label.long()
#        print(inputs.shape ,targets.shape)
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        lossd = loss.item()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        if batch_idx%50==0:
            print('Epoch=%d (%d/%d)  Loss=%.4f   lossd=%.4f'
            % (epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1), lossd))   



if __name__ == '__main__':
    net = getattr(CNN_H,args.net)(64,32,4,0)
    net=net.cuda()
    criterion = CNN_H.Super_Resolution_Loss()

    print('start: time={}'.format(dt()))
    for epoch in range(0, 20):
        if epoch in [0,5,10,18]:
            if epoch!=0: args.lr *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
        train(net,epoch,args,train_loader)
        save_model(net, 'CNN_H_{}_{}.pth'.format(args.net,epoch))
        
    print('finish: time={}\n'.format(dt()))


    