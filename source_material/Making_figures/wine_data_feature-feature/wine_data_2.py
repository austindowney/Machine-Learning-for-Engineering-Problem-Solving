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


import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
import time as time
import sklearn as sklearn
from sklearn import linear_model
from sklearn import pipeline
from sklearn import datasets
from sklearn import multiclass
from sklearn import neural_network
import json as json
import pickle as pickle

#%% Load data 

data = sk.datasets.load_wine()

y = data.target
X = data.data
feature_names = data.feature_names

X_class_1 = X[y==0]
X_class_2 = X[y==1]
X_class_3 = X[y==2]


#  		0 - Alcohol
#  		1 - Malic acid
#  		2 - Ash
# 		3 - Alcalinity of ash  
#  		4 - Magnesium
# 		5 - Total phenols
#  		6 - Flavanoids
#  		7 - Nonflavanoid phenols
#  		8 - Proanthocyanins
# 		9 - Color intensity
#  		10 - Hue
#  		11 - OD280/OD315 of diluted wines
#  		12 - Proline


# plot some feature-feature plots

plt.figure(figsize=(6.0,4))

plt.subplot(2,2,1)
feature_x = 0
feature_y = 1
plt.plot(X_class_1[:,feature_x],X_class_1[:,feature_y],'o',label='class #1')
plt.plot(X_class_2[:,feature_x],X_class_2[:,feature_y],'s',label='class #2')
plt.plot(X_class_3[:,feature_x],X_class_3[:,feature_y],'d',label='class #3')
plt.xlabel(feature_names[feature_x]+'\n (a)')
plt.ylabel(feature_names[feature_y])

plt.subplot(2,2,2)
feature_x = 9
feature_y = 6
plt.plot(X_class_1[:,feature_x],X_class_1[:,feature_y],'o',label='class #1')
plt.plot(X_class_2[:,feature_x],X_class_2[:,feature_y],'s',label='class #2')
plt.plot(X_class_3[:,feature_x],X_class_3[:,feature_y],'d',label='class #3')
plt.xlabel(feature_names[feature_x]+'\n (b)')
plt.ylabel(feature_names[feature_y])


# plt.subplot(2,2,3)
# feature_x = 0
# feature_y = 1
# plt.plot(X_class_1[:,feature_x],X_class_1[:,feature_y],'o',label='class #1')
# plt.plot(X_class_2[:,feature_x],X_class_2[:,feature_y],'s',label='class #2')
# plt.plot(X_class_3[:,feature_x],X_class_3[:,feature_y],'d',label='class #3')
# plt.xlabel(feature_names[feature_x]+'\n (c)')
# plt.ylabel(feature_names[feature_y])

# plt.subplot(2,2,4)
# feature_x = 9
# feature_y = 6
# plt.plot(X_class_1[:,feature_x],X_class_1[:,feature_y],'o',label='class #1')
# plt.plot(X_class_2[:,feature_x],X_class_2[:,feature_y],'s',label='class #2')
# plt.plot(X_class_3[:,feature_x],X_class_3[:,feature_y],'d',label='class #3')
# plt.xlabel(feature_names[feature_x]+'\n (d)')
# plt.ylabel(feature_names[feature_y])
# plt.legend()


plt.legend()
plt.tight_layout(pad=0)
plt.savefig('feature-feature',dpi=500)








































