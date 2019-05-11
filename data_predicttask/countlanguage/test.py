import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
df = pd.read_csv('anbn_test1', header=None, index_col=None)[0]
l = df.values.tolist()
ll = list(map(lambda x: len(x[x.find('s')+1:  x.find('e')]), l))
d = defaultdict(lambda:0)
for i in ll:
    d[i] += 1
x = list(d.keys())
length = len(ll)
y = list(map(lambda x: x/length, list(d.values())))
plt.bar(x, y)
plt.show()