

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X,Y  = make_regression(n_samples=100,n_features=1,noise=20, random_state=1)

plt.scatter(X,Y)

ones = np.ones([X.shape[0],1])


X = np.hstack([ones,X])

first = np.dot(X.T,X) # dot product of these two matrices

second = np.dot(X.T,Y)

theta = np.dot(np.linalg.pinv(first),second)    # linear algebra. pseudo inverse used to generate the invserse function




plt.scatter(X[:,1],Y)

X_test = [-2,2]  # 2 different values of x in order to generate the line

Y_res = []

for x in X_test:
    Y_res.append(theta[0]+theta[1]*x)

plt.plot(X_test,Y_res,c='red')

