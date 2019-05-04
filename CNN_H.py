# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 21:36:33 2019

@author: lyfeng
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict


class Super_Resolution_Loss(nn.Module):
    def __init__(self):
        super(Super_Resolution_Loss, self).__init__()
        self.eps = 1e-6
        
    def forward(self, input, target):       
        diff = torch.add(input, -target)
        error =  diff * diff + self.eps 
        loss = torch.mean(error)       
        return loss

        
        
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss.
       used in LapSRN
    """
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss


# each layer in a dense block(the base structure in Densenet)
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,bias=False))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
     
        return torch.cat([x, new_features], 1)


def Deconv_block(feature_in, feature_out, kernel_size, stride, padding, output_padding):
   return nn.Sequential(nn.ConvTranspose2d(feature_in,feature_out, kernel_size, stride, padding, output_padding),
                             nn.BatchNorm2d(feature_out),
                             nn.PReLU(feature_out))
   
def Conv_block(feature_in, feature_out, kernel_size, stride, padding):
   return nn.Sequential(nn.Conv2d(feature_in,feature_out, kernel_size, stride, padding),
                             nn.BatchNorm2d(feature_out),
                             nn.PReLU(feature_out))
   
def Conv_block_final(feature_in, feature_out, kernel_size, stride, padding):
   return nn.Sequential(nn.Conv2d(feature_in,feature_out, kernel_size, stride, padding),
                             nn.BatchNorm2d(feature_out),
                             nn.Sigmoid())        
class Hallu_Net(nn.Module):
    
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(Hallu_Net, self).__init__() 
        # initial network
        self.features = nn.Sequential(OrderedDict([("conv0", nn.Conv2d(3,64,3,1,1)),
                                                   ("norm", nn.BatchNorm2d(64)),
                                                   ("prelu", nn.PReLU(64))]))
    
        # Feature extraction(DB) and Deconvolution and Mapping(Conv)
        for i in range(6):
            self.features.add_module("DB1_%d"%(i), _DenseLayer(num_input_features+i*growth_rate, 
                                     growth_rate, bn_size, drop_rate))        
        self.features.add_module("Deconv1", Deconv_block(256,256,3,2,1,1))
        self.features.add_module("conv1", Conv_block(256,64,1,1,0))
        
        for i in range(6):
            self.features.add_module("DB2_%d"%(i), _DenseLayer(num_input_features+i*growth_rate, 
                                     growth_rate, bn_size, drop_rate))        
        self.features.add_module("Deconv2", Deconv_block(256,256,5,2,2,1))
        self.features.add_module("conv2", Conv_block(256,64,1,1,0))

        for i in range(6):
            self.features.add_module("DB3_%d"%(i), _DenseLayer(num_input_features+i*growth_rate, 
                                     growth_rate, bn_size, drop_rate))
        self.features.add_module("Deconv3", Deconv_block(256,256,5,2,2,1))
        self.features.add_module("conv3", Conv_block(256,64,1,1,0))

        # Reconstruction structure
        for i in range(2):
            self.features.add_module("DB4_%d"%(i), _DenseLayer(num_input_features+i*growth_rate, 
                                     growth_rate, bn_size, drop_rate))
        self.features.add_module("conv4", Conv_block(128,3,5,1,2))        # should delete prelu?   the output shoud be 0~1 !!!
#        self.features.add_module("conv4",Conv_block_final(128,3,5,1,2) )        # use Sigmoid func?
  
    def forward(self, x):
        return self.features(x)


