"""
Example 6.4 SVM Regression
@author: Austin R.J. Downey
"""

import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm


plt.close('all')

#%% build the data sets
np.random.seed(2) # 2 and 6 are pretty good
m = 100
X = 6 * np.random.rand(m,1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)
y = y.ravel()

# plot the data
plt.figure()
plt.grid(True)
plt.plot(X,y,'o')
plt.xlabel('x')
plt.ylabel('y')


#%% SVM regression

svm_reg = sk.svm.SVR(kernel="rbf", degree=3, C=1, epsilon=0.8, gamma="scale")
# Try poly kernal, and different degree, C, and epsilon values
svm_reg.fit(X, y)
x1 = np.linspace(-3, 3, 100).reshape(100, 1)
y_pred = svm_reg.predict(x1)


# plot the SVR model on top of the existing data
plt.plot(x1, y_pred, "-", linewidth=2, label=r"$\hat{y}$")
plt.plot(x1, y_pred + svm_reg.epsilon, "g--",label='curb')
plt.plot(x1, y_pred - svm_reg.epsilon, "g--")
plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=100,marker='o', facecolor='none', edgecolors='gray')
plt.legend(loc="upper left")