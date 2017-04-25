# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import re

DATA_DIR = "../data"

fld = open(os.path.join(DATA_DIR, "LD2011_2014.txt"), "rb")
data = []
line_num = 0
#cid = np.random.randint(0, 370, 1)
cid = 250
for line in fld:
    if line.startswith("\"\";"):
        continue
    if line_num % 100 == 0:
        print("{:d} lines read".format(line_num))
    cols = [float(re.sub(",", ".", x)) for x in 
            line.strip().split(";")[1:]]
    data.append(cols[cid])
    line_num += 1
fld.close()

NUM_ENTRIES = 1000
plt.plot(range(NUM_ENTRIES), data[0:NUM_ENTRIES])
plt.ylabel("electricity consumption")
plt.xlabel("time (1pt = 15 mins)")
plt.show()

np.save(os.path.join(DATA_DIR, "LD_250.npy"), np.array(data))
