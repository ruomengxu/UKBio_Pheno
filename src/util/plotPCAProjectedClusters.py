#! /usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/PCAProjectedClusters/part-00000')

pheno1 = df[df['cluster'] == 0]
pheno2 = df[df['cluster'] == 1]
pheno3 = df[df['cluster'] == 2]
pheno4 = df[df['cluster'] == 3]

ax = pheno1.plot(kind='scatter', x='x', y='y', color='DarkBlue', label='Phenotype 1')
pheno2.plot(kind='scatter', x='x', y='y', color='DarkGreen', label='Phenotype 2', ax=ax)
pheno3.plot(kind='scatter', x='x', y='y', color='DarkRed', label='Phenotype 3', ax=ax)
pheno4.plot(kind='scatter', x='x', y='y', color='yellow', label='Phenotype 4', ax=ax)

plt.title('PCA Projected Clusters')
plt.savefig("data/PCAProjectedClusters/pcaProjectedClusters.png")
