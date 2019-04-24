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
        qir_size = self.hidden_size + self.input_size + self.read_size
        state_size = qir_size // 2
        self.fc_state = nn.Linear(qir_size, state_size).to(self.device)
        linear_init_(self.fc_state)
        self.rnn = nn.GRUCell(state_size, self.hidden_size).to(self.device)
        gru_init_(self.rnn)
        self.fc_nargs = nn.Linear(state_size, self.n_args).to(self.device)
        linear_init_(self.fc_nargs)
        self.fc_v1 = nn.Linear(state_size, self.read_size).to(self.device)
        linear_init_(self.fc_v1)
        self.fc_v2 = nn.Linear(state_size, self.read_size).to(self.device)
        linear_init_(self.fc_v2)
        self.sigmoid = self.sigmoid.apply
        self.tanh = nn.Tanh()
    def init(self):
        hidden_shape = (self.batch_size, self.hidden_size)
        self.hidden = torch.zeros(hidden_shape).to(self.device)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, r):
        qir = torch.cat([self.hidden, x, r], dim=1)
        #output = self.tanh(self.fc_o(self.hidden))
        last_state = self.tanh(self.fc_state(qir))
        v1 = self.tanh(self.fc_v1(last_state))
        v2 = self.tanh(self.fc_v2(last_state))
        nargs = self.sigmoid(self.fc_nargs(last_state))
        self.hidden = self.rnn(last_state, self.hidden)
        instructions = torch.split(nargs, list(np.ones(self.n_args, dtype=np.int32)), dim=1)
        
        return self.hidden, v1, v2, instructions

    
