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
from stackrnn.initialization import rnn_init_, linear_init_
class PHLSTM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.weight_ih = torch.zeros([self.input_size*4, self.hidden_size], dtype=torch.float32, requires_grad=True)
        self.weight_hh = torch.zeros([self.hidden_size*4, self.hidden_size], dtype=torch.float32, requires_grad=True)
        self.bias_ih = torch.zeros([self.hidden_size*4], dtype=torch.float32, requires_grad=True)
        self.bias_hh = torch.zeros([self.hidden_size*4], dtype=torch.float32, requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, hc):
        h = hc[0]
        c = hc[1]
        
class RNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        if self.model_name == 'LSTM':
            self.cell = self.cell_class(self.input_size+self.hidden_size, self.hidden_size)
        else:
            self.cell = self.cell_class(self.input_size, self.hidden_size)
        self.hidden = None
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.nonlinear = nn.Sigmoid()
        if self.initialization:
            rnn_init_(self.cell)
            linear_init_(self.linear)
        self.cell.to(self.device)
        self.linear.to(self.device)
    def init(self):
        if self.model_name == 'GRU' or self.model_name == 'SRN':
            self.hidden = torch.zeros(self.batch_size, self.hidden_size)
        elif self.model_name == 'LSTM':
            self.hidden = (torch.zeros(self.batch_size, self.hidden_size).to(self.device), 
                           torch.zeros(self.batch_size, self.hidden_size).to(self.device))
    def forward(self, x):
        timestep = x.shape[1]
        outputs = list()
        for i in range(timestep):
            
            if self.model_name == 'LSTM':
                self.hidden = self.cell(torch.cat([x[:, i, :], self.hidden[1]], 1), self.hidden)
                tmp = self.hidden[0]
            else:
                self.hidden = self.cell(x[:, i, :], self.hidden)
                tmp = self.hidden
            output = self.nonlinear(self.linear(tmp))
            for scb in self.callback:
                scb.step_cb(self, self.hidden)
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
        self.optim = RMSprop(self.model.parameters(), lr=self.lr, momentum=0.9)
        
    def init(self):
        self.model.init(self.batch_size)
    
    def perbatch(self, xs, ys, bn=-1, istraining=True):
        batch_loss = 0
        total = 0
        correct = 0
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
        for bcb in self.callback:
            bcb.batch_cb(self.model)
        return batch_loss.item(), correct, total


