import torch
import torch.nn as nn
from torch.optim import RMSprop
from stackrnn.task import Task
from stackrnn.initialization import rnn_init_, linear_init_
class StackRNN(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        self.cell = self.cell_class(config_dict).to(self.device)
        self.o_linear = nn.Linear(self.hidden_size+self.read_size+1, self.output_size).to(self.device)
        self.nl = nn.Sigmoid()
        
        self.read = None
        self.hidden = None
        
        if self.initialization:
            linear_init_(self.o_linear)
    def forward(self, x):
        steps = x.shape[1]
        outputs = list()
        for i in range(steps):
            self.hidden, self.read = self.cell(x[:, i, :], self.hidden, self.read)
            if self.model_name.endswith('lstm'):
                onlyh = self.hidden[0]
            else:
                onlyh = self.hidden
            output = self.nl(self.o_linear(torch.cat([onlyh, self.read, self.cell.stack._actual], 1)))
            outputs.append(torch.unsqueeze(output, 1))
            for scb in self.callback:
                scb.step_cb(self.cell, x[:, i, :], output, self.read, self.hidden)
        outputs = torch.cat(outputs, 1)
        
            
        return outputs        
    
    def init(self):
        self.read = torch.zeros(self.batch_size, self.read_size).to(self.device)
        if self.model_name.endswith('lstm'):
            self.hidden = (torch.zeros(self.batch_size, self.hidden_size).to(self.device), 
                           torch.zeros(self.batch_size, self.hidden_size).to(self.device))
        else:
            self.hidden = torch.zeros((self.batch_size, self.hidden_size)).to(self.device)
        
        self.cell.stack._values = list()
        self.cell.stack._S = list()
        self.cell.stack._actual = torch.zeros(self.batch_size, 1).to(self.device)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)

class MultiStackRNN(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        self.cell = self.cell_class(config_dict).to(self.device)
        self.o_linear = nn.Linear(self.hidden_size+self.read_size*2+2, self.output_size).to(self.device)
        self.nl = nn.Sigmoid()
        
        self.read = None
        self.hidden = None
        
        if self.initialization:
            linear_init_(self.o_linear)
    def forward(self, x):
        steps = x.shape[1]
        outputs = list()
        for i in range(steps):
            self.hidden, self.read = self.cell(x[:, i, :], self.hidden, self.read)
            if self.model_name.endswith('lstm'):
                onlyh = self.hidden[0]
            else:
                onlyh = self.hidden
            output = self.nl(self.o_linear(torch.cat([onlyh, self.read, self.cell.stack0._actual, self.cell.stack1._actual], 1)))
            outputs.append(torch.unsqueeze(output, 1))
            for scb in self.callback:
                scb.step_cb(self.cell, x[:, i, :], output, self.read, self.hidden)
        outputs = torch.cat(outputs, 1)
        
            
        return outputs        
    
    def init(self):
        self.read = torch.zeros(self.batch_size, self.read_size*2).to(self.device)
        if self.model_name.endswith('lstm'):
            self.hidden = (torch.zeros(self.batch_size, self.hidden_size).to(self.device), 
                           torch.zeros(self.batch_size, self.hidden_size).to(self.device))
        else:
            self.hidden = torch.zeros((self.batch_size, self.hidden_size)).to(self.device)
        
        self.cell.stack0._values = list()
        self.cell.stack0._S = list()
        self.cell.stack0._actual = torch.zeros(self.batch_size, 1).to(self.device)

        self.cell.stack1._values = list()
        self.cell.stack1._S = list()
        self.cell.stack1._actual = torch.zeros(self.batch_size, 1).to(self.device)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
        
class StackRNNTask(Task):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.cel = nn.MSELoss(reduction="sum")
        self.optim = RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
    def perbatch(self, xs, ys, bn, istraining):
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
        ytosave = ys
        yp = yp.view(-1, self.output_size)
        ys = ys.view(-1, self.output_size)
        batch_loss = self.cel(yp, ys)
        if self.addstackloss:
            stack_loss = self.cel(self.model.cell.stack._actual, torch.zeros(self.batch_size, 1))
            batch_loss = (1 - self.alpha) * batch_loss + self.alpha * stack_loss
        if istraining:
            self.optim.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
            self.optim.step()
        if self.verbose:
            print("Train batch %d Loss: %f Accuracy: %f" % (bn, batch_loss / total, correct / total))
        for bcb in self.callback:
            bcb.batch_cb(self.model, ytosave)
        return batch_loss.item(), correct, total

class MultiStackRNNTask(Task):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.cel = nn.MSELoss(reduction="sum")
        self.optim = RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
    def perbatch(self, xs, ys, bn, istraining):
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
        ytosave = ys
        yp = yp.view(-1, self.output_size)
        ys = ys.view(-1, self.output_size)
        batch_loss = self.cel(yp, ys)
        if self.addstackloss:
            stack0_loss = self.cel(self.model.cell.stack0._actual, torch.zeros(self.batch_size, 1))
            stack1_loss = self.cel(self.model.cell.stack1._actual, torch.zeros(self.batch_size, 1))
            batch_loss = (1 - self.alpha) * batch_loss + self.alpha * (stack0_loss + stack1_loss)
        if istraining:
            self.optim.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
            self.optim.step()
        if self.verbose:
            print("Train batch %d Loss: %f Accuracy: %f" % (bn, batch_loss / total, correct / total))
        for bcb in self.callback:
            bcb.batch_cb(self.model, ytosave)
        return batch_loss.item(), correct, total

