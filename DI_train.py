# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:23:11 2019

@author: lyfeng
"""

import torch,os
import CNN_R,CNN_H
from CASIAWebFace import CASIAWebFace_dataset
from torch.autograd import Variable
import argparse
import torch.optim as optim
import torch.nn as nn
import datetime
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net_r', default='sphere36a', type=str)
parser.add_argument('--net_h', default='Hallu_Net', type=str)
parser.add_argument('--weights_r', default='./weights/CNN_R_sphere36a_19.pth', help='weights of the trained net_r')
parser.add_argument('--weights_h', default='./weights/CNN_H_Hallu_Net_19.pth', help='weights of the trained net_h')
parser.add_argument('--alpha', default=8, type=int, help='weights of L_SR & L_SI')
#parser.add_argument('--dataset', default='../../dataset/face/casia/casia.zip', type=str)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
#parser.add_argument('--bs_r', default=256, type=int, help='')
parser.add_argument('--bs', default=32, type=int, help='')
parser.add_argument('--data_root',default='../datasets/CASIA-WebFace-aligned')
parser.add_argument('--file_root',default='../datasets/casia_landmark.txt')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
#cudnn.benchmark = True

transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
])
dataset = CASIAWebFace_dataset(args.data_root, args.file_root, transform=transform, downsample=True)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=False)
 

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


class SuperIdentityLoss(nn.Module):
    def __init__(self):
        super(SuperIdentityLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):   # X,Y are feats
        diff = torch.add(L2Norm()(X), -L2Norm()(Y))
        error = diff * diff + self.eps
        loss = torch.mean(error)     # wrong here!
        return loss

# Domain-Integrated Training
def train(net_r,net_h,epoch,args,train_loader):
    net_r.train()
    net_h.train()
    train_loss = 0
    correct_I_SR = 0
    correct_I_HR = 0
    total = 0
    batch_idx = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        """ get the training data and label """
        I_LR,I_HR,label = data     # downsampled 1/8 
        if I_HR is None: break       
        I_LR = I_LR.float()   
        I_HR = I_HR.float()     
        label = label.long()
        if use_cuda: I_LR, I_HR, label = I_LR.cuda(), I_HR.cuda(), label.cuda()   
        I_LR, I_HR, label = Variable(I_LR), Variable(I_HR), Variable(label) # bs=128
        
        """ train the recognition model"""
        I_SR = net_h(I_LR)      # I_SR
        net_r.feature = False   # output prob!
        outputs_I_SR = net_r(I_SR)
        outputs_I_HR = net_r(I_HR)        
        loss_r1 = criterion_r(outputs_I_SR, label)  # bs=128 
        loss_r2 = criterion_r(outputs_I_HR, label)  # bs=128. totally bs=256 for CNN_R
        loss_r = loss_r1+loss_r2
        optimizer_r.zero_grad()
        loss_r.backward(retain_graph=True)    
        optimizer_r.step()
           
        """ train the hallucination model"""                        
        loss_h1 = criterion_h(I_SR, I_HR)  # I_SR = net_h(I_LR)   bs=128 for CNN_H
        net_r.feature = True               # output feats!
        I_SR_feats = net_r(I_SR)           # norm ! 
        I_HR_feats = net_r(I_HR)           # norm ! 
        loss_h2 = criterion_si(I_SR_feats,I_HR_feats)   
        loss_h = loss_h1 + args.alpha*loss_h2   # alpha!
        optimizer_h.zero_grad()
        loss_h.backward()
        optimizer_h.step()

        """ the recognition model training info   """
#        train_loss += loss.item()
        outputs_I_SR = outputs_I_SR[0] # 0=cos_theta 1=phi_theta
        outputs_I_HR = outputs_I_HR[0]
        _, pre_I_SR = torch.max(outputs_I_SR.data, 1)
        _, pre_I_HR = torch.max(outputs_I_HR.data, 1)        
        total += label.size(0)
        correct_I_SR += pre_I_SR.eq(label.data).cpu().sum()
        correct_I_HR += pre_I_HR.eq(label.data).cpu().sum()

        if batch_idx%50==0:
            # print the cnn_r info
            print('Epoch %d (%d/%d)  loss_r=%.4f = loss_r1=%.4f + loss_r2=%.4f  Acc(I_SR)=%.4f%% (%d/%d)  Acc(I_HR)=%.4f%% (%d/%d)  lamb=%.2f  it=%d'
            % (epoch, batch_idx, len(train_loader), loss_r.item(), loss_r1.item(), loss_r2.item(),
               100.0*correct_I_SR/total, correct_I_SR, total, 100.0*correct_I_HR/total, correct_I_HR, total,
               criterion_r.lamb, criterion_r.it)) 
#            
            # print the cnn_h info              
            print('                   loss_h=%.4f = loss_h1=%.4f + loss_h2=%.4f'
            % (loss_h.item(), loss_h1.item(), (args.alpha*loss_h2).item()))

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)     
      
if __name__ == '__main__':
    net_r = getattr(CNN_R,args.net_r)().cuda()
    net_r.load_state_dict(torch.load(args.weights_r))   
    criterion_r = CNN_R.AngleLoss()    # L_FR Recognition loss
    optimizer_r = optim.SGD(net_r.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
      
    net_h = getattr(CNN_H,args.net_h)(64,32,4,0).cuda()
    net_h.load_state_dict(torch.load(args.weights_h))   
    criterion_h = CNN_H.Super_Resolution_Loss()  # L_SR super-resolution loss
    optimizer_h = optim.SGD(net_h.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
 
    criterion_si = SuperIdentityLoss()      # L_SIsuper-identity loss
    
    print('start: time={}'.format(dt()))
    for epoch in range(0, 20):
#        print("Epoch---{:d}".format(epoch+1))
        if epoch in [0,6,10,15]:
            if epoch!=0: args.lr *= 0.1
            optimizer_r = optim.SGD(net_r.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer_h = optim.SGD(net_h.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        train(net_r,net_h,epoch,args,train_loader)
        save_model(net_r, 'FT_{}_{}.pth'.format(args.net_r,epoch))  # FT:finetune
        save_model(net_h, 'FT_{}_{}.pth'.format(args.net_h,epoch))   
      
    print('finish: time={}\n'.format(dt()))
