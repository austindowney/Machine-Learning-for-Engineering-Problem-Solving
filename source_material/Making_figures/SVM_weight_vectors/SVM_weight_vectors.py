#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open-source notes/
books. This common header is used to set the fonts and format.

Header file last updated May 16, 2024
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl


# set default fonts and plot colors
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
# I don't think I need this next line as its set to 'stixsans' above. 
plt.rcParams.update({'mathtext.fontset': 'custom'}) 
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Load special modules as needed


import matplotlib.pyplot as plt

#%% build and plot the figures


x1 = np.linspace(-3, 3, 200)

plt.figure(figsize=(6.5, 2.25))
plt.subplot(121)
w=1
b=0

y = w * x1 + b
m = 1 / w
plt.plot(x1, y)
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')
plt.plot([m, m], [0, 1], "k--", linewidth=2)
plt.plot([-m, -m], [0, -1], "k--", linewidth=2)
plt.plot([-m, m], [0, 0], "-o", linewidth=3)
plt.grid(True)
plt.xlim([-3,3])
plt.xlabel("$x_1$")
plt.ylabel("$w_1 x_1$  ")
plt.title("$w_1 = {}$".format(w))

ax2 = plt.subplot(122)


w=0.5
b=0
y = w * x1 + b
m = 1 / w
plt.plot(x1, y)
plt.axhline(y=0, color='gray')
plt.axvline(x=0, color='gray')
plt.plot([m, m], [0, 1], "k--", linewidth=2)
plt.plot([-m, -m], [0, -1], "k--", linewidth=2)
plt.plot([-m, m], [0, 0], "-o", linewidth=3)
plt.grid(True)
plt.xlim([-3,3])
plt.xlabel("$x_1$")
plt.title("$w_1 = {}$".format(w))

plt.tight_layout()
ax2.set_yticklabels([])
plt.savefig("SVM_weight_vectors",dpi=300)
plt.savefig("SVM_weight_vectors.pdf")




