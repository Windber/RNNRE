import torch.nn as nn
from stackrnn.neuralcontroller import RNNController
from stackrnn.neuralstack import NeuralStack
import torch
class StackRNNCell(nn.Module):

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        
        self._u = None
        self._s1 = None
        self._s2 = None
        self._v1 = None
        self._v2 = None


        self.controller = RNNController(config_dict)
        self.stack = NeuralStack(config_dict)
        

        

    def forward(self, inp, h, r):
        
        hidden, self._v1, self._v2, self._s1, self._u= \
        self.controller(inp, h, r)
        
        self._s2 = self._s1.clone()
        
        read = self.stack(self._u, 
                          self._s1, self._s2, 
                          self._v1, self._v2)
    
        return hidden, read
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
if __name__ == "__main__":
    pass


