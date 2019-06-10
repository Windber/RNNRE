import random
import numpy as np
import os
import pandas as pd
import re
class Palidrome:
    class Grammar:
        def __init__(self, V, T, S, P):
            self.V = V
            self.T = T
            self.S = S
            self.P = P
    def __init__(self, V, T, S, P, prob, name, diction, classes):
        self.Grammar = Palidrome.Grammar(V, T, S, P)
        self.prob = prob
        self.name = name
        self.diction = diction
        self.classes = classes
    def __str__(self):
        return self.name
    def generate(self, minlen, maxlen):
        mindepth = (minlen - 3) // 2
        maxdepth = (maxlen - 3) // 2
        depth = np.random.randint(mindepth+1, maxdepth+1)
        sl = [self.diction[i] for i in np.random.randint(0, len(self.diction.keys()), depth)]
        sl = ''.join(sl)
        s = sl + 'z' + sl[::-1]
        return s
    def accept(self, s):
        depth = 0
        length = len(s) + 2
        stack = list()
        sl = list(s)
        output = ''
        # 11 1 14 26  
        shex = hex(self.classes['e'] + self.classes['a'] + self.classes['b'] + self.classes['c'] + self.classes['d'])
        bhex = hex(self.classes['z'] + self.classes['a'] + self.classes['b'] + self.classes['c'] + self.classes['d'])
        ehex = hex(self.classes['e'])
        z = s.find('z')
        target = bhex * z
        for i in range(z-1, -1, -1):
            target += hex(self.classes[s[i]])
        target += ehex
        target = shex + target + ehex
        feature = 's' + s + 'e'
        length = len(feature) 
        depth = z
        return feature, target, length, depth
        
    def pad(self, f, l, p):
        length = len(f)
        ehex = hex(self.classes['e'])
        feature = f + 'e' * (p - length)
        label = l + ehex * (p - length)
        return feature, label

import sys
if __name__ == "__main__":
    propab = list(map(lambda x: max(0.01, min(0.99, x)), [random.random() for i in range(10)]))
    second = True
    iter = 10000 if len(sys.argv) == 1 else int(sys.argv[1])

    expectdepth = [256, 512]
    d2 = Palidrome(V=None,
                T=None,
                S=None,
                P=None,
                prob=None,
                name="palindrome",
                diction={0: "a",
                        1: "b",
                        2: "c",
                        3: "d"
                        },
                classes={'e': 1, 'z': 2, 'a': 4, 'b': 8, 'c': 16, 'd': 32}
                )
    g = [d2]
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
            s = g[i].generate(expectdepth[0], expectdepth[1])
            if s != 'secret':
                feature, predict, length, depth = g[i].accept(s)
                need = True
                if length >= 0 and length <= 32:
                    hook = _test1
                    padding = 32
                elif length > 32 and length <= 64:
                    hook = _test2
                    padding = 64
                elif length > 64 and length <= 128:
                    hook = _test3
                    padding = 128
                elif length > 128 and length <= 256:
                    hook = _test4
                    padding = 256
                elif length > 256 and length <= 512:
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
