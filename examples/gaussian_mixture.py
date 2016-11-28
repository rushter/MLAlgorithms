import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mla.kmeans import KMeans
from mla.gaussian_mixture import GaussianMixture

random.seed(1)
np.random.seed(6)


def make_clusters(skew=True, *arg, **kwargs):
    X, y = datasets.make_blobs(*arg, **kwargs)
    if skew:
        nrow = X.shape[1]
        for i in np.unique(y):
            X[y == i] = X[y == i].dot(np.random.random((nrow, nrow)) - 0.5)
    return X, y


def KMeans_and_GMM(K):
    COLOR = 'bgrcmyk'

    X, y = make_clusters(skew=True, n_samples=1500, centers=K)
    _, axes = plt.subplots(1, 3)

    # Ground Truth
    axes[0].scatter(X[:, 0], X[:, 1], c=[COLOR[int(assignment)] for assignment in y])
    axes[0].set_title("Ground Truth")

    # KMeans
    kmeans = KMeans(K=K, init='++')
    kmeans.fit(X)
    y_kmeans = kmeans.predict()
    c_kmeans = np.array(kmeans.centroids)
    axes[1].scatter(X[:, 0], X[:, 1], c=[COLOR[int(assignment)] for assignment in y_kmeans])
    axes[1].scatter(c_kmeans[:, 0], c_kmeans[:, 1], c=COLOR[:K], marker="o", s=500)
    axes[1].set_title("KMeans")

    # Gaussian Mixture
    gmm = GaussianMixture(K=K, init='kmeans')
    gmm.fit(X)
    axes[2].set_title("Gaussian Mixture")
    gmm.plot(ax=axes[2])


if __name__ == "__main__":
    KMeans_and_GMM(4)
