'''
@author: lenovo
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import RMSprop
import pandas as pd
import time
import sys
from stackrnn.task import Task
from stackrnn.initialization import rnn_init_, linear_init_
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.zeros([self.input_size*4, self.hidden_size], dtype=torch.float32, requires_grad=True))
        self.weight_hh = nn.Parameter(torch.zeros([self.hidden_size*4, self.hidden_size], dtype=torch.float32, requires_grad=True))
        self.bias_ih = nn.Parameter(torch.zeros([self.hidden_size*4], dtype=torch.float32, requires_grad=True))
        self.bias_hh = nn.Parameter(torch.zeros([self.hidden_size*4], dtype=torch.float32, requires_grad=True))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, x, hc):
        h = hc[0]
        c = hc[1]
        i = self.sigmoid(torch.mm(x, self.weight_ih[self.input_size * 0:self.input_size * 1, :])
                          + self.bias_ih[self.hidden_size * 0:self.hidden_size * 1]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 0:self.hidden_size * 1, :])
                          + self.bias_hh[self.hidden_size * 0:self.hidden_size * 1]
                          )
        
        f = self.sigmoid(torch.mm(x, self.weight_ih[self.input_size * 1:self.input_size * 2, :])
                          + self.bias_ih[self.hidden_size * 1:self.hidden_size * 2]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 1:self.hidden_size * 2, :])
                          + self.bias_hh[self.hidden_size * 1:self.hidden_size * 2]
                          )
        g = self.tanh(torch.mm(x, self.weight_ih[self.input_size * 2:self.input_size * 3, :])
                          + self.bias_ih[self.hidden_size * 2:self.hidden_size * 3]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 2:self.hidden_size * 3, :])
                          + self.bias_hh[self.hidden_size * 2:self.hidden_size * 3]
                          )
        ct = f * c + i * g
        o = self.sigmoid(torch.mm(x, self.weight_ih[self.input_size * 3:self.input_size * 4, :])
                          + self.bias_ih[self.hidden_size * 3:self.hidden_size * 4]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 3:self.hidden_size * 4, :])
                          + self.bias_hh[self.hidden_size * 3:self.hidden_size * 4]
                          )

        ht = o * self.tanh(ct)
        return ht, ct
class PHLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.zeros([self.input_size*4, self.hidden_size], dtype=torch.float32, requires_grad=True))
        self.weight_hh = nn.Parameter(torch.zeros([self.hidden_size*4, self.hidden_size], dtype=torch.float32, requires_grad=True))
        self.weight_ch = nn.Parameter(torch.zeros([self.hidden_size*3, self.hidden_size], dtype=torch.float32, requires_grad=True))
        init.orthogonal_(self.weight_ch.data)
        self.bias_ih = nn.Parameter(torch.zeros([self.hidden_size*4], dtype=torch.float32, requires_grad=True))
        self.bias_hh = nn.Parameter(torch.zeros([self.hidden_size*4], dtype=torch.float32, requires_grad=True))
        self.bias_ch = nn.Parameter(torch.zeros([self.hidden_size*3], dtype=torch.float32, requires_grad=True))
        init.uniform_(self.bias_ch.data, 0, 0)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def init(self):
        rnn_init_(self)
        amp = 2
        self.bias_ih.data.add_(torch.tensor([-amp, -amp, -amp, amp, amp, amp, 0, 0, 0, -amp, -amp, -amp], dtype=torch.float32))

    def forward(self, x, hc):
        h = hc[0]
        c = hc[1]
        i = self.sigmoid(torch.mm(x, self.weight_ih[self.input_size * 0:self.input_size * 1, :])
                          + self.bias_ih[self.hidden_size * 0:self.hidden_size * 1]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 0:self.hidden_size * 1, :])
                          + self.bias_hh[self.hidden_size * 0:self.hidden_size * 1]
                         + torch.mm(c, self.weight_ch[self.hidden_size * 0:self.hidden_size * 1, :])
                          + self.bias_ch[self.hidden_size * 0:self.hidden_size * 1]
                        
                          )
        
        f = self.sigmoid(torch.mm(x, self.weight_ih[self.input_size * 1:self.input_size * 2, :])
                          + self.bias_ih[self.hidden_size * 1:self.hidden_size * 2]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 1:self.hidden_size * 2, :])
                          + self.bias_hh[self.hidden_size * 1:self.hidden_size * 2]
                         + torch.mm(c, self.weight_ch[self.hidden_size * 1:self.hidden_size * 2, :])
                          + self.bias_ch[self.hidden_size * 1:self.hidden_size * 2]
                          )
        g = self.tanh(torch.mm(x, self.weight_ih[self.input_size * 2:self.input_size * 3, :])
                          + self.bias_ih[self.hidden_size * 2:self.hidden_size * 3]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 2:self.hidden_size * 3, :])
                          + self.bias_hh[self.hidden_size * 2:self.hidden_size * 3]
                          )
        ct = f * c + i * g
        o = self.sigmoid(torch.mm(x, self.weight_ih[self.input_size * 3:self.input_size * 4, :])
                          + self.bias_ih[self.hidden_size * 3:self.hidden_size * 4]
                         + torch.mm(h, self.weight_hh[self.hidden_size * 3:self.hidden_size * 4, :])
                          + self.bias_hh[self.hidden_size * 3:self.hidden_size * 4]
                         + torch.mm(ct, self.weight_ch[self.hidden_size * 2:self.hidden_size * 3, :])
                          + self.bias_ch[self.hidden_size * 2:self.hidden_size * 3]
                          )

        ht = o * self.tanh(ct)
        return ht, ct
class RNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        if self.model_name == 'LSTM':
            self.cell = self.cell_class(self.input_size, self.hidden_size)
        else:
            self.cell = self.cell_class(self.input_size, self.hidden_size)
        self.hidden = None
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.nonlinear = nn.Sigmoid()
        if self.initialization:
            if hasattr(self.cell, 'init'):
                self.cell.init()
            else:
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
                self.hidden = self.cell(x[:, i, :], self.hidden)
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


