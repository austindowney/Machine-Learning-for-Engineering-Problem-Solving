#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import IPython as IP
IP.get_ipython().magic('reset -sf')

#%% import modules and set default fonts and colors

"""
Default plot formatting code for Austin Downey's series of open source notes/
books. This common header is used to set the fonts and format.

Header file last updated March 10, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

# set default fonts and plot colors
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
 'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman', 
 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'}) # I don't think I need this as its set to 'stixsans' above.
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
## End of plot formatting code

plt.close('all')

#%% Load the modules needed for this code. 


import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import datasets
import pandas as pd
import seaborn as sns

plt.close('all')



#%% load data


ames = sk.datasets.fetch_openml(name="house_prices", as_frame=True)



# boston = sk.datasets.load_boston()



#Created the dataframe
ames_1 = pd.DataFrame(ames.data, columns = ames.feature_names)
drop_features = ['Id','MSSubClass','LandContour','Utilities','LotConfig',
                 'YearRemodAdd','MoSold','YrSold','PoolArea','3SsnPorch',
                 'ScreenPorch','MiscVal','GarageYrBlt','MasVnrArea','BsmtFinSF2',
                 'LowQualFinSF','WoodDeckSF','OpenPorchSF','EnclosedPorch',
                 'Fireplaces','BsmtFinSF1','BsmtHalfBath']
for i in range(len(drop_features)):
    ames_1.drop(drop_features[i], axis=1, inplace=True)
correlation_matrix = ames_1.corr().round(2)




# # build the plot
sns.heatmap(data=correlation_matrix, cmap="viridis",annot=False,cbar_kws={'label': 'correlation index'})
plt.tight_layout()
plt.savefig('Ames_housing_correlation_matirx',dpi=500)

































