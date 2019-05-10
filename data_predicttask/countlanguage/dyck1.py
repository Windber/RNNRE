import random
import numpy as np
import os
import pandas as pd
import re
class Dyck1:
    class Grammar:
        def __init__(self, V, T, S, P):
            self.V = V
            self.T = T
            self.S = S
            self.P = P
    def __init__(self, V, T, S, P, prob, name, diction):
        self.Grammar = Dyck1.Grammar(V, T, S, P)
        self.prob = prob
        self.name = name
        self.diction = diction
        self.classes = {'e': 1, '(': 2, ')': 4}
    def __str__(self):
        return self.name
    def generate(self):
        p = random.choice(self.prob)
        pattern = "[" + "".join(self.Grammar.V) + "]"
        prog = re.compile(pattern)
        cur = self.Grammar.S
        while prog.search(cur):
            mat = prog.search(cur)
            replace = mat.group(0)
            rselect = np.random.choice([0, 1], p=[1-p, p])
            rselect = rselect if len(self.Grammar.P[replace][rselect]) > 0 else 1- rselect
            replace = self.Grammar.P[replace][rselect][random.randint(0, len(self.Grammar.P[replace][rselect]) - 1)]
            cur = cur[:mat.start(0)] + replace + cur[mat.end(0):]
        return cur
    def accept(self, s):
        depth = 0
        length = len(s) + 2
        stack = list()
        sl = list(s)
        output = ''
        # 3 1 6 
        shex = hex(self.classes['('] + self.classes['e'])
        ehex = hex(self.classes['e'])
        lhex = hex(self.classes['('] + self.classes[')'])
        rhex = [hex(self.classes['('] + self.classes[')']), hex(self.classes['('] + self.classes['e']) ]
        for c in sl:
            if stack:
                if stack[-1] == "(" and c == ")":
                    stack.pop()
                else:
                    stack.append(c)
            else:
                stack.append(c)
            depth = max(depth, len(stack))
            if c == '(':
                output += lhex
            else:
                if stack:
                    output += rhex[1]
                else:
                    output += rhex[0]
        output = shex + output + ehex
        feature = 's' + s + 'e'
        return feature, output, length, depth
        
    def pad(self, f, l, p):
        length = len(f)
        ehex = hex(self.classes['e'])
        feature = f + 'e' * (p - length)
        label = l + ehex * (p - length)
        return feature, label
propab = [15/32, 31/64, 63/128, 127/256]
d1 = Dyck1(V=["S", "A", "B", "C"],
            T=["(", ")"],
            S="S",
            P={"S": [[], ["AS", "A"]],
               "A": [["C"], ["B"]],
               "B": [[], ["(S)"]],
               "C": [["()"], []]
               },
            prob=propab,
            name="dyck1",
            diction={0: "(",
                    1: ")",
                    }
            )
  
second = True
iter = 1000
if __name__ == "__main__":   
    g = [d1]
    for i in range(1):
        _test1 = set()
        _test2 = set()
        _test3 = set()
        _test4 = set()
        _test5 = set()
        hook = None
        padding = None
        count = 0
        while count < iter:
            s = g[i].generate()
            feature, predict, length, depth = g[i].accept(s)
            need = True
            if depth >=0 and depth<=8 and length >= 0 and length <= 32:
                hook = _test1
                padding = 32
            elif depth >8 and depth<=16 and length > 32 and length <= 64:
                hook = _test2
                padding = 64
            elif depth >16 and depth<=32 and length > 64 and length <= 128:
                hook = _test3
                padding = 128
            elif depth >32 and depth<=64 and length > 128 and length <= 256:
                hook = _test4
                padding = 256
            elif depth >64 and depth<=128 and length > 256 and length <= 512:
                hook = _test5
                padding = 512
            else:
                need = False
            if need:
                feature, predict = g[i].pad(feature, predict, padding)
                hook.add((feature, predict, length, depth))    
                count += 1            
        if second: 
            if os.stat(str(g[i]) + "_test1").st_size != 0:   
                o_test1 = pd.read_csv(str(g[i]) + "_test1", header=None, index_col=None, dtype={1: str})
                ol_test1 = o_test1.values.tolist()
                oll_test1 = set(list(map(tuple, ol_test1)))
                _test1 = oll_test1.union(_test1)
            if os.stat(str(g[i]) + "_test2").st_size != 0:   
                o_test2 = pd.read_csv(str(g[i]) + "_test2", header=None, index_col=None, dtype={1: str})
                ol_test2 = o_test2.values.tolist()
                oll_test2 = set(list(map(tuple, ol_test2)))
                _test2 = oll_test2.union(_test2)
            if os.stat(str(g[i]) + "_test3").st_size != 0:   
                o_test3 = pd.read_csv(str(g[i]) + "_test3", header=None, index_col=None, dtype={1: str})
                ol_test3 = o_test3.values.tolist()
                oll_test3 = set(list(map(tuple, ol_test3)))
                _test3 = oll_test3.union(_test3)
            if os.stat(str(g[i]) + "_test4").st_size != 0:   
                o_test4 = pd.read_csv(str(g[i]) + "_test4", header=None, index_col=None, dtype={1: str})
                ol_test4 = o_test4.values.tolist()
                oll_test4 = set(list(map(tuple, ol_test4)))
                _test4 = oll_test4.union(_test4)
            if os.stat(str(g[i]) + "_test5").st_size != 0:   
                o_test5 = pd.read_csv(str(g[i]) + "_test5", header=None, index_col=None, dtype={1: str})
                ol_test5 = o_test5.values.tolist()
                oll_test5 = set(list(map(tuple, ol_test5)))
                _test5 = oll_test5.union(_test5)
        df08 = pd.DataFrame(list(_test1))
        df08.to_csv(str(g[i]) + "_test1", header=None, index=False)
        df816 = pd.DataFrame(list(_test2))
        df816.to_csv(str(g[i]) + "_test2", header=None, index=False)
        df1632 = pd.DataFrame(list(_test3))
        df1632.to_csv(str(g[i]) + "_test3", header=None, index=False)
        df3264 = pd.DataFrame(list(_test4))
        df3264.to_csv(str(g[i]) + "_test4", header=None, index=False)
        df64128 = pd.DataFrame(list(_test5))
        df64128.to_csv(str(g[i]) + "_test5", header=None, index=False)
        print(g[i], len(_test1), len(_test2), len(_test3), len(_test4), len(_test5))
