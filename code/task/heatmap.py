#import torch
import h5py
import seaborn
f = h5py.File("../stackrnn/sdata/hotmap", "r")
data = f['batch0']
sample = data[0]

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(sample, annot=True, linewidths=.5, ax=ax)
plt.show()