import random
import numpy as np
import os
import pandas as pd
import re
class Dyck2:
    class Grammar:
        def __init__(self, V, T, S, P):
            self.V = V
            self.T = T
            self.S = S
            self.P = P
    def __init__(self, V, T, S, P, prob, name, diction):
        self.Grammar = Dyck2.Grammar(V, T, S, P)
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
#     def generate(self):
#         pairs = random.randint(0, 128)
#         cur = "()"
#         length = 0
#         for i in range(pairs):
#             length += 2
#             insert = random.randint(0, length)
#             cur = cur[:insert] + "()" + cur[insert:]
#         return cur
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
    def accept(self, s):
        depth = 0
        stack = list()
        sl = list(s)
        output = list()
        for c in sl:
            if stack:
                if ( stack[-1] == "[" and c == "]" ) or ( stack[-1] == "(" and c == ")" ):
                    stack.pop()
                else:
                    stack.append(c)
            else:
                stack.append(c)
            depth = max(depth, len(stack))
            if stack:
                output.append("0")
            else:
                output.append("1")
        return "".join(output), "0" if stack else "1", depth

propab = [1/2, 3/8, 7/16, 15/32]
#propab = list(map(lambda x: 1 - x, propab))
negmaxlen = 256
second = True
iter = 1000
d2 = Dyck2(V=["S", "R", "B", "T"],
            T=["(", ")", "[", "]"],
            S="S",
            P={"S": [[], ["R", "RS"]],
               "R": [["T"], ["B"]],
               "B": [[], ["[S]", "(S)"]],
               "T": [["()", "[]"], []]
               },
            prob=propab,
            name="Dyck2",
            diction={0: "(",
                    1: ")",
                    2: "[",
                    3: "]"
                    }
            )
  
if __name__ == "__main__":   
    g = [d2]
    for i in range(1):
        l08 = set()
        l816 = set()
        l1632 = set()
        l32inf = set()
        hook = None
        padding = None
        for _ in range(iter):
            s = g[i].generate()
            serial, label, depth = g[i].accept(s)
            s = "s" + s + "e"
            serial = "0" + serial + serial[-1]
            length = len(s)
            need = False
            if depth<=8 and length <= 32:
                hook = l08
                padding = 32
                need = True
            elif depth<=16 and length <= 256 and depth > 8:
                hook = l816
                padding = 256
                need = True
            elif depth<=32 and length <= 256 and depth > 16:
                hook = l1632
                padding = 256
                need = True
            elif length<= 512 and depth > 32:
                hook = l32inf
                padding = 512
                need = True
            if need:
                s = s + "#" * (padding - length)
                serial = serial + serial[-1] * (padding - length)
                hook.add((length, depth, label, s, serial))
                s = g[i].negative(s[1: s.find("e")], negmaxlen)
                if s:
                    serial, label, depth = g[i].accept(s)
                    s = "s" + s + "e"
                    serial = "0" + serial + serial[-1]
                    length = len(s)
                    need = False
                    if depth<=8 and length <= 32:
                        hook = l08
                        padding = 32
                        need = True
                    elif depth<=16 and length <= 256 and depth > 8:
                        hook = l816
                        padding = 256
                        need = True
                    elif depth<=32 and length <= 256 and depth > 16:
                        hook = l1632
                        padding = 256
                        need = True
                    elif length<= 512 and depth > 32:
                        hook = l32inf
                        padding = 512
                        need = True
                    if need:
                        s = s + "#" * (padding - length)
                        serial = serial + serial[-1] * (padding - length)
                        hook.add((length, depth, label, s, serial))                
                        
        if second: 
            if os.stat(str(g[i]) + "l08").st_size != 0:   
                ol08 = pd.read_csv(str(g[i]) + "l08", header=None, index_col=None, dtype={1: str})
                oll08 = ol08.values.tolist()
                olll08 = set(list(map(tuple, oll08)))
                l08 = olll08.union(l08)
            if os.stat(str(g[i]) + "l816").st_size != 0:   
                ol816 = pd.read_csv(str(g[i]) + "l816", header=None, index_col=None, dtype={1: str})
                oll816 = ol816.values.tolist()
                olll816 = set(list(map(tuple, oll816)))
                l816 = olll816.union(l816)
            if os.stat(str(g[i]) + "l1632").st_size != 0:   
                ol1632 = pd.read_csv(str(g[i]) + "l1632", header=None, index_col=None, dtype={1: str})
                oll1632 = ol1632.values.tolist()
                olll1632 = set(list(map(tuple, oll1632)))
                l1632 = olll1632.union(l1632)
            if os.stat(str(g[i]) + "l32inf").st_size != 0:   
                ol32inf = pd.read_csv(str(g[i]) + "l32inf", header=None, index_col=None, dtype={1: str})
                oll32inf = ol32inf.values.tolist()
                olll32inf = set(list(map(tuple, oll32inf)))
                l32inf = olll32inf.union(l32inf)
        df08 = pd.DataFrame(list(l08))
        df08.to_csv(str(g[i]) + "l08", header=None, index=False)
        df816 = pd.DataFrame(list(l816))
        df816.to_csv(str(g[i]) + "l816", header=None, index=False)
        df1632 = pd.DataFrame(list(l1632))
        df1632.to_csv(str(g[i]) + "l1632", header=None, index=False)
        df32inf = pd.DataFrame(list(l32inf))
        df32inf.to_csv(str(g[i]) + "l32inf", header=None, index=False)
        print(g[i], len(l08), len(l816), len(l1632), len(l32inf))
        
        np = list()
        dflist = [df08, df816, df1632, df32inf]
        for df in dflist:
            if len(df) > 0:
                p = sum(map(int, df[2].values.tolist()))
                n = len(df) - p
                np.append((n, p))
        for i in np:
            print(i)