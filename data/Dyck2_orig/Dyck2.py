'''
@author: lenovo
'''
import re
import numpy as np
import pandas as pd
import os
class Dyck2:
    def __init__(self, probability=0.5):
        self.gr = {"S": ["R", "RS"],
                   "R": ["T", "B"],
                   "B": ["[S]", "(S)"],
                   "T": ["()", "[]"]
                   }
        self.start = "S"
        self.reg = re.compile("[SRBT]")
        
        self.probability = probability 
        self.probabilitys = [self.probability, 1-self.probability]
    def generate(self):
        s = self.start
        while self.reg.search(s):
            index = self.reg.search(s).start()
            if s[index] == "S" or s[index] == "R":
                tmp = np.random.choice([0, 1], p=self.probabilitys)
            else:
                tmp = np.random.randint(0, 2)
            s = s[:index] + self.gr[s[index]][tmp] + s[index+1:]
        return s
#         length = np.random.randint(1, 16)
#         oral = np.random.randint(0, 2, length)
#         d = {0: "()",
#              1: "[]"
#         }
#         l = [d[i] for i in oral]
#         s = "".join(l)
#         return s
    def setprobability(self, p):
        self.probability = p
        self.probabilitys = [self.probability, 1-self.probability]
    
    def accept(self, s):
        depth = 0
        stack = list()
        sl = list(s)
        for c in sl:
            if stack:
                if ( stack[-1] == "[" and c == "]" ) or ( stack[-1] == "(" and c == ")" ):
                    stack.pop()
                else:
                    stack.append(c)
            else:
                stack.append(c)
            depth = max(depth, len(stack))
        return (True if len(stack) == 0 else False, depth)

if __name__ == "__main__":
    second = True
    d = Dyck2(0.55)
    s08 = set()
    s816 = set()
    s1632 = set()
    s3264 = set()
    s64inf = set()
    hook = None
    iter = 10000
    for i in range(iter):
        s = d.generate()
        pos, depth = d.accept(s)
        length = len(s)
        if depth < 8:
            hook = s08
        elif depth < 16:
            hook = s816
        elif depth < 32:
            hook = s1632
        elif depth < 64:
            hook = s3264
        else:
            hook = s64inf
        hook.add((length, depth, pos, s))
    if second: 
        if os.stat("s08").st_size != 0:   
            os08 = pd.read_csv("s08", header=None, index_col=None)
            l08 = os08.values.tolist()
            os08 = set(list(map(tuple, l08)))
            s08 = os08.union(s08)
        if os.stat("s816").st_size != 0:
            os816 = pd.read_csv("s816", header=None, index_col=None)
            l816 = os816.values.tolist()
            os816 = set(list(map(tuple, l816)))
            s816 = os816.union(s816)
        if os.stat("s1632").st_size != 0:
            os1632 = pd.read_csv("s1632", header=None, index_col=None)
            l1632 = os1632.values.tolist()
            os1632 = set(list(map(tuple, l1632)))
            s1632 = os1632.union(s1632)
        if os.stat("s3264").st_size != 0:
            os3264 = pd.read_csv("s3264", header=None, index_col=None)
            l3264 = os3264.values.tolist()
            os3264 = set(list(map(tuple, l3264)))
            s3264 = os3264.union(s3264)
        if os.stat("s64inf").st_size != 0:
            os64inf = pd.read_csv("s64inf", header=None, index_col=None)
            l64inf = os64inf.values.tolist()
            os64inf = set(list(map(tuple, l64inf)))
            s64inf = os64inf.union(s64inf)
    df08 = pd.DataFrame(list(s08))
    df08.to_csv("s08", header=None, index=False)
    df816 = pd.DataFrame(list(s816))
    df816.to_csv("s816", header=None, index=False)
    df1632 = pd.DataFrame(list(s1632))
    df1632.to_csv("s1632", header=None, index=False)
    df3264 = pd.DataFrame(list(s3264))
    df3264.to_csv("s3264", header=None, index=False)
    df64inf = pd.DataFrame(list(s64inf))
    df64inf.to_csv("s64inf", header=None, index=False)
    print(len(s08), len(s816), len(s1632), len(s3264), len(s64inf))
    
    #l816 = list(s816)
    