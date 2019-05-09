'''
@author: lenovo
'''
import torch
import torch.nn as nn
from torch.optim import RMSprop
import pandas as pd
import time
import sys
from stackrnn.task import Task
class RNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.cell = self.cellclass(self.input_size, self.hidden_size)
        self.hidden = None
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.nonlinear = nn.Sigmoid()
        if self.initialization:
            nn.init.xavier_uniform_(self.linear.weight.data)
            nn.init.uniform_(self.linear.bias.data, 0, 0)
            nn.init.xavier_uniform_(self.cell.weight_ih.data)
            nn.init.orthogonal_(self.cell.weight_hh.data)
            nn.init.uniform_(self.cell.bias_ih.data, 0, 0)
            nn.init.uniform_(self.cell.bias_hh.data, 0, 0)
        
        self.cell.to(self.device)
        self.linear.to(self.device)
    def init(self):
        if self.model_name == 'GRU' or self.model_name == 'RNN':
            self.hidden = torch.zeros(self.batch_size, self.hidden_size)
        elif self.model_name == 'LSTM':
            self.hidden = (torch.zeros(self.batch_size, self.hidden_size).to(self.device), 
                           torch.zeros(self.batch_size, self.hidden_size).to(self.device))
    def forward(self, x):
        timestep = x.shape[1]
        outputs = list()
        for i in range(timestep):
            self.hidden = self.cell(x[:, i, :], self.hidden)
            output = self.nonlinear(self.linear(self.hidden[0]))
            outputs.append(torch.unsqueeze(output, 1))
        outputs = torch.cat(outputs, 1)
        return outputs
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
class RNNTask(Task):
    def __init__(self, params):
        super().__init__(params)
        self.cel = nn.MSELoss(reduction="sum")
        self.optim = RMSprop(self.model.parameters(), lr=self.lr)
        
    def init(self):
        self.model.init(self.batch_size)
    
    def perbatch(self, xs, ys, bn=-1, istraining=True):
        batch_loss = 0
        total = 0
        correct = 0
        steps = xs.shape[1]
        batch_size = xs.shape[0]
        self.model.init()
        yp = self.model(xs)
        total = batch_size
        
        yc = yp.clone().detach()
        yc.apply_(lambda x: 1. if x >=0.5 else 0.)
        yc = yc!=ys
        yc = torch.sum(yc, dim=(1, 2))
        yc.apply_(lambda x: 1. if x == 0. else 0.)
        correct = torch.sum(yc).item()
        
        yp = yp.view(-1, self.output_size)
        ys = ys.view(-1, self.output_size)
        batch_loss = self.cel(yp, ys)
         
        
        if istraining:
            self.optim.zero_grad()
            batch_loss.backward()
            self.optim.step()
        if self.verbose:
            print("Train batch %d Loss: %f Accuracy: %f" % (bn, batch_loss / total, correct / total))
        return batch_loss.item(), correct, total


