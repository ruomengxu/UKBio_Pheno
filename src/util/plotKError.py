#! /usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/KMeansKError/part-00000')

df.sort_values(by='k').set_index('k').plot()

plt.title('K vs WSSSE')
plt.savefig("data/KMeansKError/kErrorPlot.png")
