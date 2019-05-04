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
from PIL import Image
import numpy as np
from scipy.signal import convolve2d


parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net_r', default='sphere36a', type=str)
parser.add_argument('--net_h', default='Hallu_Net', type=str)
parser.add_argument('--weights_r', default='./weights/CNN_R_sphere36a_19.pth', help='weights of the trained net_r')
parser.add_argument('--weights_h', default='./weights/CNN_H_Hallu_Net_19.pth', help='weights of the trained net_h')
parser.add_argument('--alpha', default=8, type=int, help='weights of L_SR & L_SI')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
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


def test(net_r,net_h,args,train_loader):
    net_r.eval()
    net_h.eval()

    
    for batch_idx, data in enumerate(train_loader):
        
        """ get the training data and label """
        I_LR,I_HR,label = data    
        if I_HR is None: break       
        I_LR = I_LR.float()   
        I_HR = I_HR.float()     
        label = label.long()
        if use_cuda: I_LR, I_HR, label = I_LR.cuda(), I_HR.cuda(), label.cuda()   
        I_LR, I_HR, label = Variable(I_LR), Variable(I_HR), Variable(label) # bs=128
        
        """ test the recognition model"""
        import matplotlib.pyplot as plt
        I_SR = net_h(I_LR)      # I_SR
        n = 10
        I_SR = I_SR[n]
        I_HR = I_HR[n]
        I_LR = I_LR[n]
        

        I_HR = transforms.ToPILImage()((I_HR*0.5+0.5).cpu())

        plt.subplot(131)
        I_SR = transforms.ToPILImage()((I_SR*0.5+0.5).cpu())        
        plt.title('SR img(PSNR:{:.4f} SSIM:{:.4f})'.format(calpsnr(np.array(I_SR),np.array(I_HR)),
                          compute_ssim(np.array(I_SR)[:,:,0],np.array(I_HR)[:,:,0])))
        plt.imshow(I_SR)
        
        plt.subplot(132)
        plt.title('HR img')
        plt.imshow(I_HR)
                
        plt.subplot(133)
        I_LR = transforms.ToPILImage()((I_LR*0.5+0.5).cpu())
        I_LR = I_LR.resize((I_LR.size[0]*8, I_LR.size[1]*8), resample=Image.BICUBIC)
        plt.title('BICUBIC img(PSNR:{:.4f} SSIM:{:.4f})'.format(calpsnr(np.array(I_LR),np.array(I_HR)),
                          compute_ssim(np.array(I_LR)[:,:,0],np.array(I_HR)[:,:,0])))       
        plt.imshow(I_LR)      
        plt.show()
        break


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def calpsnr(im1, im2):
    #print(im1.shape)
    #print(im2.shape)
    #exit()
    diff = np.abs(im1 - im2)
    rmse = np.sqrt(np.mean(np.square(diff)))
    psnr = 20*np.log10(255/rmse)
    return psnr
      
      
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


    
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

    test(net_r,net_h,args,train_loader)

