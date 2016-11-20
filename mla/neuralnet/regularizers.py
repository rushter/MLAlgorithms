# coding:utf-8
from autograd import elementwise_grad
import numpy as np


class Regularizer(object):
    def __init__(self, C=0.01):
        self.C = C
        self._grad = elementwise_grad(self._penalty)

    def _penalty(self, weights):
        raise NotImplementedError()

    def grad(self, weights):
        return self._grad(weights)

    def __call__(self, weights):
        return self.grad(weights)


class L1(Regularizer):
    def _penalty(self, weights):
        return self.C * np.abs(weights)


class L2(Regularizer):
    def _penalty(self, weights):
        return self.C * weights ** 2


class ElasticNet(Regularizer):
    """Linear combination of L1 and L2 penalties."""
    def _penalty(self, weights):
        return 0.5 * self.C * weights ** 2 + (1.0 - self.C) * np.abs(weights)
