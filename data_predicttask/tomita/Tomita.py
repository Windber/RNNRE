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
    
    def predict(self, s, padding, pred_class={'e': 1, '0': 2, '1': 4}):
        cur = self.FSA.Q0
        l = list(s)
        output = list()
        
        tmp = 0
        for i in self.Grammar.P['A'][0]:
            tmp += pred_class[i[0]]
        tmp += pred_class['e']
        output.append(hex(tmp))
        
        while l:
            t = l.pop(0)
            cur = self.FSA.rDELTA[(cur, t)]
            tmp = 0
            for i in self.Grammar.P[cur][0]:
                tmp += pred_class[i[0]]
            tmp += pred_class['e']
            output.append(hex(tmp))
        output.append(hex(pred_class['e']))
        output = ''.join(output)
        length = len(s)
        feature = 's' + s + 'e' + 'e' * (padding - length - 2)
        label = output + hex(pred_class['e'])*(padding - length - 2)
        return feature, label
        
propab = [1/4, 1/2, 3/4, 7/8, 15/16, 31/32, 63/64, 127/128]
negmaxlen = 256
pred_class= {'e': 1, '0': 2, '1': 4}
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
     name="t1_"
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
     name="t2_"
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
     name="t3_"
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
     name="t4_"
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
     name="t5_"
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
     name="t6_"
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
     name="t7_"
     )
if __name__ == "__main__":
    
    second = True
    iter = 10000

    
    g = [g3, g5, g6, g7]
    for i in range(len(g)):
        test1 = set()
        test2 = set()
        test3 = set()
        test4 = set()
        test5 = set()

        for _ in range(iter):
            s = g[i].generate()
            length = len(s) + 2
            hook = None
            padding = None
            if length <= 32:
                hook = test1
                padding = 32
            elif length <= 64:
                hook = test2
                padding = 64
            elif length <= 128:
                hook = test3
                padding = 128
            elif length <= 256:
                hook = test4
                padding = 256
            elif length <= 512:
                hook = test5
                padding = 512
            if padding:
                feature, label = g[i].predict(s, padding)
                hook.add((feature, label))
        if second: 
            if os.stat(str(g[i]) + "test1").st_size != 0:   
                otest1 = pd.read_csv(str(g[i]) + "test1", header=None, index_col=None, dtype={1: str})
                oltest1 = otest1.values.tolist()
                olltest1 = set(list(map(tuple, oltest1)))
                test1 = olltest1.union(test1)
            if os.stat(str(g[i]) + "test2").st_size != 0:   
                otest2 = pd.read_csv(str(g[i]) + "test2", header=None, index_col=None, dtype={1: str})
                oltest2 = otest2.values.tolist()
                olltest2 = set(list(map(tuple, oltest2)))
                test2 = olltest2.union(test2)
            if os.stat(str(g[i]) + "test3").st_size != 0:   
                otest3 = pd.read_csv(str(g[i]) + "test3", header=None, index_col=None, dtype={1: str})
                oltest3 = otest3.values.tolist()
                olltest3 = set(list(map(tuple, oltest3)))
                test3 = olltest3.union(test3)
            if os.stat(str(g[i]) + "test4").st_size != 0:   
                otest4 = pd.read_csv(str(g[i]) + "test4", header=None, index_col=None, dtype={1: str})
                oltest4 = otest4.values.tolist()
                olltest4 = set(list(map(tuple, oltest4)))
                test4 = olltest4.union(test4)
            if os.stat(str(g[i]) + "test5").st_size != 0:   
                otest5 = pd.read_csv(str(g[i]) + "test5", header=None, index_col=None, dtype={1: str})
                oltest5 = otest5.values.tolist()
                olltest5 = set(list(map(tuple, oltest5)))
                test5 = olltest5.union(test5)
        dftest1 = pd.DataFrame(list(test1))
        dftest1.to_csv(str(g[i]) + "test1", header=None, index=False)
        dftest2 = pd.DataFrame(list(test2))
        dftest2.to_csv(str(g[i]) + "test2", header=None, index=False)
        dftest3 = pd.DataFrame(list(test3))
        dftest3.to_csv(str(g[i]) + "test3", header=None, index=False)
        dftest4 = pd.DataFrame(list(test4))
        dftest4.to_csv(str(g[i]) + "test4", header=None, index=False)
        dftest5 = pd.DataFrame(list(test5))
        dftest5.to_csv(str(g[i]) + "test5", header=None, index=False)
        print(g[i], len(test1), len(test2), len(test3), len(test4), len(test5))