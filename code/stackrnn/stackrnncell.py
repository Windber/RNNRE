import torch.nn as nn
from stackrnn.neuralcontroller import RNNController, MultiRNNController
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

class MultiStackRNNCell(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict

        self.controller = MultiRNNController(config_dict)
        self.stack0 = NeuralStack(config_dict)
        self.stack1 = NeuralStack(config_dict)
    
    def forward(self, inp, h, r):
        hidden, self._v01, self._v02, self._s01, self._u0, self._v11, self._v12, self._s11, self._u1 = \
        self.controller(inp, h, r)
        
        self._s02 = self._s01.clone()
        self._s12 = self._s11.clone()
        
        read0 = self.stack0(self._u0, 
                          self._s01, self._s02, 
                          self._v01, self._v02)
        read1 = self.stack1(self._u1, 
                          self._s11, self._s12, 
                          self._v11, self._v12)        
        read = torch.cat([read0, read1], 1)
        return hidden, read
if __name__ == "__main__":
    pass


