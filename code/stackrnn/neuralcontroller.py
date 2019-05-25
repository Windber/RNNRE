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
            if self.addubias:
                self.fc_u.bias.data.add_(torch.tensor(self.ubias, dtype=torch.float32))
            if self.customization:
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
                elif self.data_name == 'dyck1':
                    self.fc_v2.weight.data.add_(torch.tensor([[0, 0, 0, 0, 0, 0, apm, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_v1.weight.data.add_(torch.tensor([[0, 0, apm, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, apm, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_u.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, apm, apm, apm],
                                                             dtype=torch.float32))
                    self.fc_s1.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, -apm, apm, -apm],
                                                             dtype=torch.float32))
                elif self.data_name == 'dyck2':
                    self.fc_v2.weight.data.add_(torch.tensor([[0, 0, 0, 0, 0, 0, apm, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0, 0, 0, apm, 0]], dtype=torch.float32))
                    self.fc_v1.weight.data.add_(torch.tensor([[0, 0, apm, 0, 0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, apm, 0, 0, 0, 0, 0, 0]], dtype=torch.float32))
                    self.fc_u.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, apm, apm, apm, apm, apm],
                                                             dtype=torch.float32))
                    self.fc_s1.weight.data.add_(torch.tensor([0, 0, 0, 0, apm, -apm, apm, -apm, apm, -apm],
                                                             dtype=torch.float32))
                    
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, h, r):
        ri = torch.cat([r, x], dim=1)
        if self.model_name.endswith('lstm'):
            onlyh = h[0]
        else:
            onlyh = h
        qri = torch.cat([onlyh, ri], dim=1)
        
        s1 = self.sigmoid(self.fc_s1(qri))
        u = self.sigmoid(self.fc_u(qri))
        v1 = self.tanh(self.fc_v1(qri))
        v2 = self.tanh(self.fc_v2(qri))
        hidden = self.rnn(ri, h)
        
        return hidden, v1, v2, s1, u

class MultiRNNController(nn.Module):
    def __init__(self, config_dict):
        
        super().__init__()
        self.params = config_dict
        ir_size = self.input_size + self.read_size * 2
        qir_size = self.hidden_size + ir_size
        self.rnn = self.controller_cell_class(ir_size, self.hidden_size).to(self.device)
        self.fc_s01 = nn.Linear(qir_size, 1).to(self.device)
        self.fc_u0 = nn.Linear(qir_size, 1).to(self.device)
        self.fc_v01 = nn.Linear(qir_size, self.read_size).to(self.device)
        self.fc_v02 = nn.Linear(qir_size, self.read_size).to(self.device)

        self.fc_s11 = nn.Linear(qir_size, 1).to(self.device)
        self.fc_u1 = nn.Linear(qir_size, 1).to(self.device)
        self.fc_v11 = nn.Linear(qir_size, self.read_size).to(self.device)
        self.fc_v12 = nn.Linear(qir_size, self.read_size).to(self.device)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        if self.initialization:
            linear_init_(self.fc_v02)
            linear_init_(self.fc_v01)
            linear_init_(self.fc_s01)
            linear_init_(self.fc_u0)
            
            linear_init_(self.fc_v12)
            linear_init_(self.fc_v11)
            linear_init_(self.fc_s11)
            linear_init_(self.fc_u1)
            if self.addubias:
                self.fc_u0.bias.data.add_(torch.tensor(self.ubias, dtype=torch.float32))
                self.fc_u1.bias.data.add_(torch.tensor(-self.ubias, dtype=torch.float32))
            rnn_init_(self.rnn)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, h, r):
        ri = torch.cat([r, x], dim=1)
        if self.model_name.endswith('lstm'):
            onlyh = h[0]
        else:
            onlyh = h
        qri = torch.cat([onlyh, ri], dim=1)
        
        s01 = self.sigmoid(self.fc_s01(qri))
        u0 = self.sigmoid(self.fc_u0(qri))
        v01 = self.tanh(self.fc_v01(qri))
        v02 = self.tanh(self.fc_v02(qri))

        s11 = self.sigmoid(self.fc_s11(qri))
        u1 = self.sigmoid(self.fc_u1(qri))
        v11 = self.tanh(self.fc_v11(qri))
        v12 = self.tanh(self.fc_v12(qri))
        
        hidden = self.rnn(ri, h)
        
        return hidden, v01, v02, s01, u0, v11, v12, s11, u1

    
    
