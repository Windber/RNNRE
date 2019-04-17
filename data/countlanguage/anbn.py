import random
import numpy as np
import os
import pandas as pd
import re
class Anbn:
    class Grammar:
        def __init__(self, V, T, S, P):
            self.V = V
            self.T = T
            self.S = S
            self.P = P
    def __init__(self, V, T, S, P, prob, name, diction):
        self.Grammar = Anbn.Grammar(V, T, S, P)
        self.prob = prob
        self.name = name
        self.diction = diction
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
    def negative(self, posstring, maxlength):
        iteration = 0
        if posstring:
            ran = posstring
        else:
            ran = np.random.randint(0, len(self.diction.keys()), random.randint(1, maxlength))
            ran = "".join(list(map(lambda x: self.diction[x], list(ran))))
        length = len(ran)
        while self.accept(ran)[1] == "1":
            minindex = random.randint(0, length-1)
            maxindex = random.randint(minindex+1, length)
            tmp = list(ran[minindex: maxindex])
            random.shuffle(tmp)
            ran = ran[:minindex] + "".join(tmp) + ran[maxindex:]
            iteration += 1
            if iteration > 10:
                return None
#                 ran = np.random.randint(0, 2, length)
#                 ran = "".join(list(map(str, list(ran))))
                #print("in trap: %s" % (ran))
        return ran
    def accept_sub(self, s):
        length = len(s)
        right = "a" * int(length / 2) + "b" * int(length / 2)
        if s == right:
            return True
        else:
            return False
    def accept(self, s):
        output = list()
        length = len(s)
        result = None
        for i in range(length):
            if self.accept_sub(s[: i+1]):
                output.append("1")
                result = "1"
            else:
                output.append("0")
                result = "0"
        return "".join(output), result

propab = [1/2, 3/4, 7/8]
propab = list(map(lambda x: x, propab))
negmaxlen = 32
anbn = Anbn(V=["A", "B"],
            T=["a", "b"],
            S="A",
            P={"A": [["ab"], ["aAB"]],
               "B": [["b"], []]
               },
            prob=propab,
            name="Anbn",
            diction={0: "a",
                        1: "b"}
            )
second = True
iter = 10000

g = [anbn]
for i in range(1):
    l032 = set()
    l3264 = set()
    l64128 = set()
    l128256 = set()
    hook = None
    padding = None
    for _ in range(iter):
        s = g[i].generate()
        serial, label = g[i].accept(s)
        sn = g[i].negative(s, negmaxlen)
        s = "s" + s + "e"
        serial = "0" + serial + serial[-1]
        if sn:
            serialn, labeln = g[i].accept(sn)
            sn = "s" + sn + "e"
            serialn = "0" + serialn + serialn[-1]
        length = len(s)
        if length <= 32:
            hook = l032
            padding = 32
        elif length <= 64:
            hook = l3264
            padding = 64
        elif length <= 128:
            hook = l64128
            padding = 128
        else:
            hook = l128256
            padding = 256
        if length <= 256:
            s = s + "#" * (padding - length)
            serial = serial + serial[-1] * (padding - length)
            hook.add((length, label, s, serial))
            if sn:
                sn = sn + "#" * (padding - length)
                serialn = serialn + serialn[-1] * (padding - length)
                hook.add((length, labeln, sn, serialn))
                 
             
    if second: 
        if os.stat(str(g[i]) + "l032").st_size != 0:   
            ol032 = pd.read_csv(str(g[i]) + "l032", header=None, index_col=None, dtype={1: str})
            oll032 = ol032.values.tolist()
            olll032 = set(list(map(tuple, oll032)))
            l032 = olll032.union(l032)
        if os.stat(str(g[i]) + "l3264").st_size != 0:   
            ol3264 = pd.read_csv(str(g[i]) + "l3264", header=None, index_col=None, dtype={1: str})
            oll3264 = ol3264.values.tolist()
            olll3264 = set(list(map(tuple, oll3264)))
            l3264 = olll3264.union(l3264)
        if os.stat(str(g[i]) + "l64128").st_size != 0:   
            ol64128 = pd.read_csv(str(g[i]) + "l64128", header=None, index_col=None, dtype={1: str})
            oll64128 = ol64128.values.tolist()
            olll64128 = set(list(map(tuple, oll64128)))
            l64128 = olll64128.union(l64128)
        if os.stat(str(g[i]) + "l128256").st_size != 0:   
            ol128256 = pd.read_csv(str(g[i]) + "l128256", header=None, index_col=None, dtype={1: str})
            oll128256 = ol128256.values.tolist()
            olll128256 = set(list(map(tuple, oll128256)))
            l128256 = olll128256.union(l128256)
    df032 = pd.DataFrame(list(l032))
    df032.to_csv(str(g[i]) + "l032", header=None, index=False)
    df3264 = pd.DataFrame(list(l3264))
    df3264.to_csv(str(g[i]) + "l3264", header=None, index=False)
    df64128 = pd.DataFrame(list(l64128))
    df64128.to_csv(str(g[i]) + "l64128", header=None, index=False)
    df128256 = pd.DataFrame(list(l128256))
    df128256.to_csv(str(g[i]) + "l128256", header=None, index=False)
    print(g[i], len(l032), len(l3264), len(l64128), len(l128256))