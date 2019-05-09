import torch.nn as nn
import numpy as np
import torch
from stackrnn.initialization import gru_init_, linear_init_
class GRUController(nn.Module):
    def __init__(self, config_dict):
        
        super().__init__()
        self.params = config_dict
        ir_size = self.input_size + self.read_size
        qir_size = self.hidden_size + ir_size
        self.rnn = nn.GRUCell(ir_size, self.hidden_size).to(self.device)
        self.fc_nargs = nn.Linear(qir_size, self.n_args).to(self.device)
        self.fc_v1 = nn.Linear(qir_size, self.read_size).to(self.device)
        self.fc_v2 = nn.Linear(qir_size, self.read_size).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        
        if self.initialization:
            linear_init_(self.fc_v2)
            linear_init_(self.fc_v1)
            linear_init_(self.fc_nargs)
            gru_init_(self.rnn)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, h, r):
        ir = torch.cat([x, r], dim=1)
        qir = torch.cat([h, ir], dim=1)
        
        nargs = self.sigmoid(self.fc_nargs(qir))
        v1 = self.tanh(self.fc_v1(qir))
        v2 = self.tanh(self.fc_v2(qir))
        hidden = self.rnn(ir, h)
        
        instructions = torch.split(nargs, list(np.ones(self.n_args, dtype=np.int32)), dim=1)
        
        return hidden, v1, v2, instructions

    
