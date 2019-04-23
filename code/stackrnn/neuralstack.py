import torch.nn as nn
from abc import ABCMeta, abstractmethod
import torch
class NeuralMemory(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.params = kwargs
        self._values = None
        self._S = None
        self._actual = None
        
    def forward(self, u, d1, d2, v1, v2, r=None):
        self.push(d1, d2, v1, v2)
        readcontent = self.read(u)
        self.pop(u)
        return readcontent
    def init(self):
        self._values = list()
        self._S = list()
        self._actual = torch.zeros(self.batch_size, 1)
    def push(self, strength1, strength2, value1, value2):
        self._actual = self._actual + strength1 + strength2
        self._values.append(value1)
        self._S.append(strength1)
        self._values.append(value2)
        self._S.append(strength2)
        
    def read(self, u):
        summary = torch.zeros([self.batch_size, self.read_size])
        strength_used = torch.zeros(self.batch_size, 1)
        for i in self._read_indices():
            summary += self._values[i] * torch.min(self._S[i], torch.max(u - strength_used))
            strength_used += self._S[i]
        return summary
    
    def pop(self, u):
        self._actual = torch.max(torch.zeros(self.batch_size, 1), self._actual - u)
        strength_used = torch.zeros(self.batch_size, 1)
        for i in self._pop_indices():
            tmp = self._S[i]
            self._S[i] = self._S[i] - torch.min(self._S[i], torch.max(torch.zeros(self.batch_size, 1), u - strength_used))
            strength_used += tmp
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
class NeuralStack(NeuralMemory):
    def __init__(self, kwargs):
        super().__init__(kwargs)
    def _pop_indices(self):
        return list(range(len(self._S)-1, -1, -1))
    
    def _push_index(self):
        return len(self._S)
    
    def _read_indices(self):
        return list(range(len(self._S)-1, -1, -1))
    
class NeuralQueue(NeuralMemory):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        
    def _pop_indices(self):
        return list(range(0, len(self._S)))
    
    def _push_index(self):
        return len(self._S)
    
    def _read_indices(self):
        return list(range(0, len(self._S)))
if __name__ == "__main__":
    from  mystackrnn.profile import config_dyck2
    ns = NeuralStack(config_dyck2)
    print(ns)
