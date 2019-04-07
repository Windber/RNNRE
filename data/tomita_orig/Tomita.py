'''
@author: lenovo
'''
import random
import numpy as np
import os
import pandas as pd
class Tomita:
    '''
    class for determination regular grammar
    '''
    class Grammar:
        def __init__(self, V, T, S, P):
            self.V = V
            self.T = T
            self.S = S
            self.P = P
    class FSA:
        def __init__(self, Q, SIGMA, DELTA, Q0, F):
            self.Q = Q
            self.SIGMA = SIGMA
            self.DELTA = DELTA
            self.Q0 = Q0
            self.F = F
            self.rDELTA = dict()
            self.DELTAL = {k: len(self.DELTA[k]) for k in self.DELTA.keys()}
            for k in self.DELTA.keys():
                for t, v in self.DELTA[k]:
                    self.rDELTA[(k, t)] = v
    def __init__(self, V, T, S, P, Q, SIGMA, DELTA, Q0, F, propab, name):
        self.Grammar = Tomita.Grammar(V, T, S, P)
        self.FSA = Tomita.FSA(Q, SIGMA, DELTA, Q0, F)
        self.propab = propab
        self.name = name
    def __str__(self):
        return self.name
    def generate(self):
        p = random.choice(self.propab)
        cur = self.Grammar.S
        l = list()
        count = 0
        while cur:
            rselect = np.random.choice([0, 1], p=[1-p, p])
            rselect = rselect if len(self.Grammar.P[cur][rselect]) > 0 else 1- rselect
            tmp = self.Grammar.P[cur][rselect][random.randint(0, len(self.Grammar.P[cur][rselect]) - 1)]
            cur = tmp[1] if len(tmp) == 2 else None
            l.append(tmp[0])
            count += 1
        
        return "".join(l)
    def negative(self, posstring, maxlength):
        iteration = 0
        if posstring:
            ran = posstring
        else:
            ran = np.random.randint(0, 2, random.randint(1, maxlength))
            ran = "".join(list(map(str, list(ran))))
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
        cur = self.FSA.Q0
        l = list(s)
        output = list()
        while l:
            t = l.pop(0)
            #print(cur, t)
            cur = self.FSA.rDELTA[(cur, t)]
            isaccept = "1" if cur in self.FSA.F else "0"
            #print(cur, isaccept)
            output.append(isaccept)
        if cur in self.FSA.F:
            return "".join(output), "1"
        else:
            return "".join(output), "0"

if __name__ == "__main__":
    propab = [1/4, 1/2, 3/4]
    negmaxlen = 256
    second = True
    iter = 0
    g1 = Tomita(V=["A"], 
                 T=["0", "1"], 
                 S="A",
                 P={"A": [[("1")], [("1", "A")]],
                    },
                 Q=["A", "B"],
                 SIGMA=["0", "1"],
                 Q0="A",
                 F=["A"],
                 DELTA={"A": [("1", "A"), ("0", "B")],
                        "B": [("1", "B"), ("0", "B")]
                        },
                 propab=propab,
                 name="Tomita1"
                 )
    g2 = Tomita(V=["A", "B"], 
                 T=["0", "1"], 
                 S="A",
                 P={"A": [[], [("1", "B")]],
                    "B": [[("0")], [("0", "A")]]
                    },
                 Q=["A", "B", "C"],
                 SIGMA=["0", "1"],
                 Q0="A",
                 F=["A"],
                 DELTA={"A": [("1", "B"), ("0", "C")],
                        "B": [("1", "C"), ("0", "A")],
                        "C": [("0", "C"), ("1", "C")]
                        },
                 propab=propab,
                 name="Tomita2"
                 )
    g3 = Tomita(V=["A", "B", "C", "D"], 
                 T=["0", "1"], 
                 S="A",
                 P={"A": [[("0"), ("1")], [("0", "A"), ("1", "B")]],
                    "B": [[("1")], [("0", "D"), ("1", "A")]],
                    "C": [[("1")], [("0", "D"), ("1", "B")]],
                    "D": [[("0")], [("0", "C")]]
                    },
                 Q=["A", "B", "C", "D", "E"],
                 SIGMA=["0", "1"],
                 Q0="A",
                 F=["A", "B", "C"],
                 DELTA={"A": [("0", "A"), ("1", "B")],
                        "B": [("0", "D"), ("1", "A")],
                        "C": [("0", "D"), ("1", "B")],
                        "D": [("0", "C"), ("1", "E")],
                        "E": [("0", "E"), ("1", "E")]
                        },
                 propab=propab,
                 name="Tomita3"
                 )
    g4 = Tomita(V=["A", "B", "C"], 
                 T=["0", "1"], 
                 S="A",
                 P={"A": [[("0"), ("1"),], [("1", "A"), ("0", "B")]],
                    "B": [[("0"), ("1"),], [("0", "C"), ("1", "A")]],
                    "C": [[("1"),], [("1", "A")]],
                    },
                 Q=["A", "B", "C", "D"],
                 SIGMA=["0", "1"],
                 Q0="A",
                 F=["A", "B", "C"],
                 DELTA={"A": [("1", "A"), ("0", "B")],
                        "B": [("0", "C"), ("1", "A")],
                        "C": [("0", "D"), ("1", "A")],
                        "D": [("1", "D"), ("0", "D")]
                        },
                 propab=propab,
                 name="Tomita4"
                 )
    g5 = Tomita(V=["A", "B", "C", "D"], 
                 T=["0", "1"], 
                 S="A",
                 P={"A": [[], [("0", "C"), ("1", "B")]],
                    "B": [[("1")], [("0", "D"), ("1", "A")]],
                    "C": [[("0")], [("0", "A"), ("1", "D")]],
                    "D": [[], [("0", "B"), ("1", "C")]]
                    },
                 Q=["A", "B", "C", "D"],
                 SIGMA=["0", "1"],
                 Q0="A",
                 F=["A"],
                 DELTA={"A": [("0", "C"), ("1", "B")],
                        "B": [("0", "D"), ("1", "A")],
                        "C": [("0", "A"), ("1", "D")],
                        "D": [("0", "B"), ("1", "C")]
                        },
                 propab=propab,
                 name="Tomita5"
                 )
    g6 = Tomita(V=["A", "B", "C"], 
                 T=["0", "1"], 
                 S="A",
                 P={"A": [[], [("0", "C"), ("1", "B")]],
                    "B": [[("0")], [("0", "A"), ("1", "C")]],
                    "C": [[("1")], [("1", "A"), ("0", "B")]],
                    },
                 Q=["A", "B", "C"],
                 SIGMA=["0", "1"],
                 Q0="A",
                 F=["A"],
                 DELTA={"A": [("0", "C"), ("1", "B")],
                        "B": [("0", "A"), ("1", "C")],
                        "C": [("1", "A"), ("0", "B")],
                        },
                 propab=propab,
                 name="Tomita6"
                 )
    g7 = Tomita(V=["A", "B", "C", "D"], 
                 T=["0", "1"], 
                 S="A",
                 P={"A": [[("0"), ("1")], [("0", "A"), ("1", "B")]],
                    "B": [[("0"), ("1")], [("1", "B"), ("0", "C")]],
                    "C": [[("0"), ("1")], [("0", "C"), ("1", "D")]],
                    "D": [[("1")], [("1", "D")]],
                    },
                 Q=["A", "B", "C", "D", "E"],
                 SIGMA=["0", "1"],
                 Q0="A",
                 F=["A", "B", "C", "D"],
                 DELTA={"A": [("0", "A"), ("1", "B")],
                        "B": [("1", "B"), ("0", "C")],
                        "C": [("0", "C"), ("1", "D")],
                        "D": [("1", "D"), ("0", "E")],
                        "E": [("1", "E"), ("0", "E")]
                        },
                 propab=propab,
                 name="Tomita7"
                 )
    g = [g1, g2, g3, g4, g5, g6, g7]
    for i in range(7):
        l032 = set()
        l3264 = set()
        l64128 = set()
        l128256 = set()
        hook = None
        padding = None
        for _ in range(iter):
            s = g[i].generate()
            serial, label = g[i].accept(s)
            sn = g[i].negative(s, random.randint(1, 256))
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