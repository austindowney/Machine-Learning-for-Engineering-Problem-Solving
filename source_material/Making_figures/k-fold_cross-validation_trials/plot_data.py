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

#%% Load your data

results = np.load('results_1000.pickle',allow_pickle=True)

for key, val in results.items():
     exec(key + '=val')

results_500 = np.load('results_500.pickle',allow_pickle=True)

metrics_sgd_accuracy = metrics_sgd_accuracy + results_500['metrics_sgd_accuracy']
metrics_k_fold_accuracy = metrics_k_fold_accuracy + results_500['metrics_k_fold_accuracy']
metrics_sgd_precision = metrics_sgd_precision + results_500['metrics_sgd_precision']
metrics_k_fold_precision = metrics_k_fold_precision + results_500['metrics_k_fold_precision']
metrics_sgd_recall = metrics_sgd_recall + results_500['metrics_sgd_recall']
metrics_k_fold_recall = metrics_k_fold_recall + results_500['metrics_k_fold_recall']

#%% accuracy

#x_bins=35 # np.int(np.sqrt(len(metrics_sgd_accuracy))) #np.arange(0,100,1)

x_bins=np.arange(0,100,0.2)

# the histogram of the data
plt.figure(figsize=(6.5,2.1))
plt.subplot(131)
n, bins, patches = plt.hist(np.array(metrics_sgd_accuracy)*100, x_bins, density=False, alpha=0.5,label='full data',log=True)
n, bins, patches = plt.hist(np.array(metrics_k_fold_accuracy)*100, x_bins, density=False, alpha=0.5,label='k-fold',log=True)
plt.xlabel(r'accuracy (\%)')
plt.ylabel('count')
plt.legend(framealpha=1,fontsize=9)
plt.xlim(80, 100)
plt.ylim(0.7, 400)
plt.grid(True)


x_bins=np.arange(0,100,1)

plt.subplot(132)
n, bins, patches = plt.hist(np.array(metrics_sgd_precision)*100, x_bins, density=False, alpha=0.5,label='full-data',log=True)
n, bins, patches = plt.hist(np.array(metrics_k_fold_precision)*100, x_bins, density=False, alpha=0.5,label='k-fold',log=True)
plt.xlabel(r'precision (\%)')
#plt.ylabel('count')
plt.xlim(25, 100)
plt.ylim(0.7, 400)
plt.grid(True)

plt.subplot(133)
n, bins, patches = plt.hist(np.array(metrics_sgd_recall)*100, x_bins, density=False, alpha=0.5,label='full-data',log=True)
n, bins, patches = plt.hist(np.array(metrics_k_fold_recall)*100, x_bins, density=False, alpha=0.5,label='k-fold',log=True)
plt.xlabel(r'recall (\%)')
#plt.ylabel('count')
plt.xlim(25, 100)
plt.ylim(0.7, 400)
plt.grid(True)
plt.tight_layout()
plt.savefig('k-fold_cross-validation_trials.png',dpi=300)






