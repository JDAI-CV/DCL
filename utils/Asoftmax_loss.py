import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

import pdb

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 50.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target, decay=None):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        if decay is None:
            self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        else:
            self.LambdaMax *= decay 
            self.lamb = max(self.LambdaMin, self.LambdaMax) 
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

