#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np

X = np.load("../data/game-screenshots.npy")
print(X.shape)

sidx = 140
for soff in range(4):
    plt.subplot(sidx + soff + 1)
    plt.imshow(X[soff].T, cmap="gray")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()