import torch.nn as nn
from stackrnn.neuralcontroller import GRUController
from stackrnn.neuralstack import NeuralStack
from stackrnn.initialization import gru_init_, linear_init_
import torch
class StackRNNCell(nn.Module):

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        
        self._read = None
        self._u = None
        self._s1 = None
        self._s2 = None
        self._v1 = None
        self._v2 = None

        self.controller = GRUController(config_dict)
        self.struct = NeuralStack(config_dict)
        
        self.fc_o = nn.Linear(self.hidden_size + 1, self.output_size).to(self.device)
        linear_init_(self.fc_o)
        self.tanh = nn.Tanh()
        

    def init(self):
        self.struct.init()
        
        self.controller.init()

        self._read = torch.zeros(self.batch_size, self.read_size).to(self.device)
        

    def forward(self, inp):
        
        hidden, self._v1, self._v2, (self._s1, self._u)= self.controller(inp, self._read)
        
        self._s2 = self._s1.clone()
        
        self._read = self.struct(self._u, self._s1, self._s2, self._v1, self._v2)
        
        o = self.tanh(self.fc_o(torch.cat([hidden, self.struct._actual], dim=1)))
        
        return o
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
if __name__ == "__main__":
    from stackrnntask.profile import config_dyck2
    ns = NeuralStack(config_dyck2)
    gc = GRUController(config_dyck2)
    print(ns, gc)


