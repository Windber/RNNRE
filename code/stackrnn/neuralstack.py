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
        readcontent = self.pop(u)
        return readcontent
    def push(self, strength1, strength2, value1, value2):
        self._actual = self._actual + strength1 + strength2
        self._values.append(value1)
        self._S.append(strength1)
        self._values.append(value2)
        self._S.append(strength2)
        
    def pop(self, u):
        summary = torch.zeros([self.batch_size, self.read_size]).to(self.device)
        strength_used = torch.zeros(self.batch_size, 1).to(self.device)
        for i in self._read_indices():
            per_used = torch.min(self._S[i], torch.max(u - strength_used, torch.zeros(self.batch_size, 1)))
            if torch.sum(per_used.detach()).item() != 0:
                strength_used = strength_used + per_used
                summary = summary + self._values[i] * per_used
                self._S[i] = self._S[i] - per_used
                if torch.sum(strength_used.detach()) == torch.sum(u.detach()):
                    break
        self._actual = torch.max(torch.zeros(self.batch_size, 1).to(self.device), self._actual - u)
        return summary

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
    pass
