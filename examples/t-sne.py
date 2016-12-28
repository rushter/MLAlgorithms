import logging

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from mla.tsne import TSNE

logging.basicConfig(level=logging.DEBUG)

X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=0, random_state=1111,
                           n_classes=2, class_sep=2.5, )

p = TSNE(2, max_iter=500)
X = p.fit_transform(X)

colors = ['red', 'green']
for t in range(2):
    t_mask = (y == t).astype(bool)
    plt.scatter(X[t_mask, 0], X[t_mask, 1], color=colors[t])

plt.show()
