# coding:utf-8
import numpy as np

EPSILON = 10e-8


class Constraint(object):
    def clip(self, p):
        return p


class MaxNorm(object):
    def __init__(self, m=2, axis=0):
        self.axis = axis
        self.m = m

    def clip(self, p):
        norms = np.sqrt(np.sum(p ** 2, axis=self.axis))
        desired = np.clip(norms, 0, self.m)
        p = p * (desired / (EPSILON + norms))
        return p


class NonNeg(object):
    def clip(self, p):
        p[p < 0.] = 0.
        return p


class SmallNorm(object):
    def clip(self, p):
        return np.clip(p, -5, 5)


class UnitNorm(Constraint):
    def __init__(self, axis=0):
        self.axis = axis

    def clip(self, p):
        return p / (EPSILON + np.sqrt(np.sum(p ** 2, axis=self.axis)))
