import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

# the last FC layer
class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))    # weights
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)       # Initialization
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [         
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,                  # cos(4*theta) 
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)              # 每次前向传播时归一化权重
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()    # k*phi=m*theta
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1         
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))    # why neef lamb? 损失所占权重不同
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)      # 很大权重还是在cos_theta部分  
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)      # phi_theta占了一小部分，这种操作是怎么确定的？

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

    

class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
#        self.bn1 = nn.BatchNorm2d(channels)         # delete all BN layers!
        self.prelu1 = nn.PReLU(channels) 
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
#        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU(channels)

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
#        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
#        x = self.bn2(x)
        x = self.prelu2(x)

        return x + short_cut


class sphere36a(nn.Module):
    def __init__(self, classnum=10574,feature=False):
        super(sphere36a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        num_layers = 36
        assert num_layers in [20, 36, 64], 'SphereNet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 36:
            layers = [2, 4, 8, 2]
        elif num_layers == 64:
            layers = [3, 8, 16, 3]
        else: 
            raise ValueError('sphere' + str(num_layers) + " IS NOT SUPPORTED! (sphere20 or sphere64)")

        filter_list = [3, 64, 128, 256, 512]
        block = Block
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.fc1 = nn.Linear(512 * 7 * 6, 512)
#        self.last_bn = nn.BatchNorm1d(512)
        self.fc2 = AngleLinear(512,self.classnum)


    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
#        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.feature:
            return x
#        x = self.last_bn(x)
        x = self.fc2(x)

        return x
