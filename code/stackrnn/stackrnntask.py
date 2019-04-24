from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pandas as pd
import random
import time
from stackrnn.neuralcontroller import GRUController
from stackrnn.neuralstack import NeuralStack
from stackrnn.task import Task
class StackRNNTask(Task):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.loss_classify = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weight), reduction='sum')
        self.loss_stacklength = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
    def perbatch(self, bx, by, b, istraining):
        bsize = self.batch_size
        steps = bx.shape[1]
        btotal = bsize * steps
        yp = torch.zeros((bsize, steps, self.output_size)).to(self.device)
        for i in range(steps):
            outp = self.model(bx[:, i, :])
            yp[:, i, :] = outp
        
        _, yp_index = torch.topk(yp, 1, dim=2)
        yp_index = yp_index.view(yp_index.shape[0], yp_index.shape[1])
        bcorrect = torch.sum( yp_index == by).item()
        yp = yp.view(-1, 2)
        ys = by.view(-1)
        bloss = self.loss_classify(yp, ys)
        slen = self.model.struct._actual.view(-1)
        stackloss = torch.zeros(1)
        for i in range(self.batch_size):
            if ys[i].item() == 1:
                stackloss = stackloss + self.loss_stacklength(slen[i], torch.zeros(1).to(self.device))
        bloss = (1-sum(self.loss_weight)) * bloss + self.loss_weight[0] * stackloss
        if istraining:
            self.optimizer.zero_grad()
            bloss.backward()
            self.optimizer.step()
        if self.verbose:
            print("Batch %d Loss: %f Accuracy: %f" % (b, bloss / btotal, bcorrect / btotal))
        return bloss, bcorrect, btotal
    
    def perstep(self):
        pass
    

