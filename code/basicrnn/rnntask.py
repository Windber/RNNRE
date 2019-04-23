'''
@author: lenovo
'''
import torch
import torch.nn as nn
from torch.optim import RMSprop
import pandas as pd
import time
import sys
from task.myprofile import *
import tensorflow as tf
import tensorflow.keras as keras
from stackrnn.task import Task
class RNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True, )
        self.hidden = None
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.rnn.to(self.device)
        self.linear.to(self.device)
    def init(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
    def forward(self, x):
        tmp, self.hidden = self.rnn(x, self.hidden)
        output = self.linear(tmp)
        return output
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__()
class RNNTask(Task):
    def __init__(self, params):
        super().__init(params)
        self.cel = nn.CrossEntropyLoss(reduction="sum")
        self.optim = RMSprop(self.model.parameters(), lr=self.lr)
        
    def init(self):
        self.model.init(self.batch_size)
    
    def perbatch(self, xs, ys, bn=-1, istraining=True):
        batch_loss = 0
        total = 0
        correct = 0
        steps = xs.shape[1]
        batch_size = xs.shape[0]
        self.model.init(self.batch_size)
        yp = self.model(xs)
        _, yp_index = torch.topk(yp, 1, dim=2)
        total = batch_size * steps
        yp_index = yp_index.view(yp_index.shape[0], yp_index.shape[1])
        correct = torch.sum( yp_index == ys).item()
        yp = yp.view(-1, 2)
        ys = ys.view(-1)

        batch_loss = self.cel(yp, ys)

        
        if istraining:
            self.optim.zero_grad()
            batch_loss.backward()
            self.optim.step()
        if self.verbose:
            print("Train batch %d Loss: %f Accuracy: %f" % (bn, batch_loss / total, correct / total))
        return batch_loss, correct, total


