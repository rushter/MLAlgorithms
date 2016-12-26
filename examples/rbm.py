import logging

import numpy as np

from mla.rbm import RBM

logging.basicConfig(level=logging.DEBUG)


def print_curve(rbm):
    from matplotlib import pyplot as plt

    def moving_average(a, n=25):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.plot(moving_average(rbm.errors))
    plt.show()


X = np.random.uniform(0, 1, (1500, 10))
rbm = RBM(n_hidden=10, max_epochs=200, batch_size=10, learning_rate=0.1)
rbm.fit(X)
print_curve(rbm)


