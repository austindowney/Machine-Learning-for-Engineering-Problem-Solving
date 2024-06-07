#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().magic('reset -sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open source notes/
books. This common header is used to set the fonts and format.

Last updated March 10, 2024
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as PD
import scipy as sp
from scipy import interpolate
import pickle
import time
import re
import json as json
import pylab
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
plt.rcParams.update({'mathtext.fontset': 'custom'}) # I don't think I need this as its set to 'stixsans' above.
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% plot the arctan



years = [2010,2011,2012,2013,2014]
error = [28,26,15,11,6]
winner = ['NEC-UIUC','Xerox Research Centre Europe','AlexNet, University of Toronto','Clarifai','GoogleLeNet']


plt.figure(figsize=(6,3))
plt.plot(years,error,'o-')


plt.grid('on')
plt.ylabel('winning error (\%)')
plt.xticks(years)
plt.xlabel('year')
plt.ylim([3.5,32.5])
plt.xlim([2009.8,2014.78])
plt.vlines(2011.5,0,50,colors=('red'),linestyles='dashed')
plt.text(2009.95,6,'traditional ML methods',bbox=dict(facecolor='white', edgecolor='white'))
plt.text(2011.7,6,'deep learning methods',bbox=dict(facecolor='white', edgecolor='white'))
for i in range(len(winner)):
    plt.text(years[i],error[i]+2,winner[i],bbox=dict(facecolor='white', edgecolor='white'))
plt.arrow(2009.9,5,1.45,0,color='red',linewidth=3, length_includes_head=True,
          head_width=1.3, head_length=0.1,zorder=10)
plt.arrow(2011.65,5,1.45,0,color='red',linewidth=3, length_includes_head=True,
          head_width=1.3, head_length=0.1,zorder=10)
plt.tight_layout()
plt.savefig('ImageNet-competition-results',dpi=275)












































































