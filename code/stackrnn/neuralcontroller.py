import torch.nn as nn
import numpy as np
import torch
from stackrnn.initialization import gru_init_, linear_init_
class GRUController(nn.Module):
    def __init__(self, config_dict):
        
        super().__init__()
        self.hidden = None
        self.params = config_dict
        
        #controller_input_size = self.input_size + self.read_size
        ir_size = self.input_size + self.read_size
        qir_size = self.hidden_size + ir_size
        state_size = qir_size // 2
        self.fc_state = nn.Linear(qir_size, state_size).to(self.device)
        linear_init_(self.fc_state)
        self.rnn = nn.GRUCell(ir_size, self.hidden_size).to(self.device)
        gru_init_(self.rnn)
        self.fc_nargs = nn.Linear(state_size, self.n_args).to(self.device)
        linear_init_(self.fc_nargs)
        self.fc_v1 = nn.Linear(ir_size, self.read_size).to(self.device)
        linear_init_(self.fc_v1)
        self.fc_v2 = nn.Linear(ir_size, self.read_size).to(self.device)
        linear_init_(self.fc_v2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
    def init(self):
        hidden_shape = (self.batch_size, self.hidden_size)
        self.hidden = torch.zeros(hidden_shape).to(self.device)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, r):
        ir = torch.cat([x, r], dim=1)
        qir = torch.cat([self.hidden, ir], dim=1)
        
        state = self.leaky_relu(self.fc_state(qir))
        nargs = self.sigmoid(self.fc_nargs(state))
        v1 = self.tanh(self.fc_v1(ir))
        v2 = self.tanh(self.fc_v2(ir))
        self.hidden = self.rnn(ir, self.hidden)
        
        instructions = torch.split(nargs, list(np.ones(self.n_args, dtype=np.int32)), dim=1)
        
        return self.hidden, v1, v2, instructions

    
