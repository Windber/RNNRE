import torch.nn as nn
import numpy as np
import torch
from stackrnn.initialization import rnn_init_, linear_init_
class RNNController(nn.Module):
    def __init__(self, config_dict):
        
        super().__init__()
        self.params = config_dict
        ir_size = self.input_size + self.read_size
        qir_size = self.hidden_size + ir_size
        self.rnn = self.controller_cell_class(ir_size, self.hidden_size).to(self.device)
        self.fc_s1 = nn.Linear(qir_size, 1).to(self.device)
        self.fc_u = nn.Linear(qir_size, 1).to(self.device)
        self.fc_v1 = nn.Linear(qir_size, self.read_size).to(self.device)
        self.fc_v2 = nn.Linear(qir_size, self.read_size).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        
        if self.initialization:
            linear_init_(self.fc_v2)
            linear_init_(self.fc_v1)
            linear_init_(self.fc_s1)
            linear_init_(self.fc_u)
            rnn_init_(self.rnn)
            self.fc_u.bias.data.add_(torch.tensor(-1, dtype=torch.float32))
            self.fc_s1.bias.data.add_(torch.tensor(1, dtype=torch.float32))
            if self.customalization:
                apm = 5
                if self.data_name == 'anbn':
                    self.fc_v2.weight.data.add_(torch.tensor([[0, 0, 0, 0, 0, 0, apm, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_v1.weight.data.add_(torch.tensor([[0, 0, apm, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, apm, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_u.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, apm, apm, apm],
                                                             dtype=torch.float32))
                    self.fc_s1.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, -apm, apm, -apm],
                                                             dtype=torch.float32))
                    self.rnn.weight_ih.data.add_(torch.tensor([[0, 0, 0, 0, apm, 0],
                                                               [0, 0, 0, 0, 0, apm]], dtype=torch.float32))
                elif self.data_name == 'dyck1':
                    self.fc_v2.weight.data.add_(torch.tensor([[0, 0, 0, 0, 0, 0, apm, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_v1.weight.data.add_(torch.tensor([[0, 0, apm, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, apm, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_u.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, apm, apm, apm],
                                                             dtype=torch.float32))
                    self.fc_s1.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, -apm, apm, -apm],
                                                             dtype=torch.float32))
                    self.rnn.weight_ih.data.add_(torch.tensor([[0, 0, 0, 0, apm, 0],
                                                               [0, 0, 0, 0, 0, 0]], dtype=torch.float32))
                elif self.data_name == 'dyck2':
                    self.fc_v2.weight.data.add_(torch.tensor([[0, 0, 0, 0, 0, 0, apm, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0, 0, 0, apm, 0]], dtype=torch.float32))
                    self.fc_v1.weight.data.add_(torch.tensor([[0, 0, apm, 0, 0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, apm, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_u.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, apm, apm, apm, apm, apm],
                                                             dtype=torch.float32))
                    self.fc_s1.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, -apm, apm, -apm, apm, -apm],
                                                             dtype=torch.float32))
                    self.rnn.weight_ih.data.add_(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0],
                                                               [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
                    
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, h, r):
        ri = torch.cat([r, x], dim=1)
        qri = torch.cat([h, ri], dim=1)
        
        s1 = self.sigmoid(self.fc_s1(qri))
        u = self.sigmoid(self.fc_u(qri))
        v1 = self.tanh(self.fc_v1(qri))
        v2 = self.tanh(self.fc_v2(qri))
        hidden = self.rnn(ri, h)
        
        return hidden, v1, v2, s1, u

    
