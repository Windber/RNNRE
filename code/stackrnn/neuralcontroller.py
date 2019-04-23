import torch.nn as nn
import numpy as np
import torch
class GRUController(nn.Module):
    def __init__(self, config_dict):
        
        super().__init__()
        self.hidden = None
        self.params = config_dict
        
        controller_input_size = self.input_size + self.read_size
        state_size = self.hidden_size + self.input_size + self.read_size
        self.rnn = nn.GRUCell(controller_input_size, self.hidden_size)
        self.fc_nargs = nn.Linear(state_size, self.n_args)
        self.fc_v1 = nn.Linear(state_size, self.read_size)
        self.fc_v2 = nn.Linear(state_size, self.read_size)
        self.sigmoid = self.sigmoid.apply
        self.tanh = nn.Tanh()
    def init(self):
        hidden_shape = (self.batch_size, self.hidden_size)
        self.hidden = torch.zeros(hidden_shape)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, r):
        last_state = torch.cat([self.hidden, x, r], dim=1)
        #output = self.tanh(self.fc_o(self.hidden))
        v1 = self.tanh(self.fc_v1(last_state))
        v2 = self.tanh(self.fc_v2(last_state))
        nargs = self.sigmoid(self.fc_nargs(last_state))
        self.hidden = self.rnn(torch.cat([x, r], 1), self.hidden)
        instructions = torch.split(nargs, list(np.ones(self.n_args, dtype=np.int32)), dim=1)
        
        return self.hidden, v1, v2, instructions

    
