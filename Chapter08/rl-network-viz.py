# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "../data"

xs, loss_ys, wins_ys = [], [], []
fin = open(os.path.join(DATA_DIR, "rl-network-results-5100.tsv"), "rb")
lno = 0
for line in fin:
#    if lno < 1000:
#        lno += 1
#        continue
    cols = line.strip().split("\t")
    epoch = int(cols[0])
    loss = float(cols[1])
    num_wins = int(cols[2])
    xs.append(epoch)
    loss_ys.append(loss)
    wins_ys.append(num_wins)
    lno += 1
fin.close()
    
plt.subplot(211)
plt.plot(xs, loss_ys)
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("loss (MSE)")

plt.subplot(212)
plt.plot(xs, wins_ys)
plt.title("Wins")
plt.xlabel("epochs")
plt.ylabel("# wins (cumulative)")

plt.tight_layout()
plt.show()
