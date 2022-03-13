# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 13:24:30 2021

@author: merve
"""

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

X, y = make_blobs(1000, n_features=300, centers=4,
                  cluster_std=8, random_state=42)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
rand = np.random.RandomState(42)

for axi in ax.flat:
    i, j = rand.randint(X.shape[1], size=2)
    axi.scatter(X[:, i], X[:, j], c=y)
    
    from lpproj import LocalityPreservingProjection
lpp = LocalityPreservingProjection(n_components=2)

X_2D = lpp.fit_transform(X)

plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y)

plt.title("Projected from 500->2 dimensions");