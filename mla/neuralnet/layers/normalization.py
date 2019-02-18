from mla.neuralnet.layers import Layer, PhaseMixin, ParamMixin
from mla.neuralnet.parameters import Parameters
import numpy as np

"""
References:
https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
"""

class BatchNormalization(Layer, ParamMixin, PhaseMixin):
    def __init__(self, momentum=0.9, eps=1e-5, parameters=None):
        super().__init__()
        self._params = parameters
        if self._params is None:
            self._params = Parameters()
        self.momentum = momentum
        self.eps = eps
        self.ema_mean = None
        self.ema_var = None

    def setup(self, x_shape):
        self._params.setup_weights((1, x_shape[1]))

    def _forward_pass(self, X):
        gamma = self._params['W']
        beta = self._params['b']

        if self.is_testing:
            mu = self.ema_mean
            xmu = X - mu
            var = self.ema_var
            sqrtvar = np.sqrt(var + self.eps)
            ivar = 1. / sqrtvar
            xhat = xmu * ivar
            gammax = gamma * xhat
            return gammax + beta

        N, D = X.shape

        # step1: calculate mean
        mu = 1. / N * np.sum(X, axis=0)

        # step2: subtract mean vector of every trainings example
        xmu = X - mu

        # step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        # step4: calculate variance
        var = 1. / N * np.sum(sq, axis=0)

        # step5: add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + self.eps)

        # step6: invert sqrtwar
        ivar = 1. / sqrtvar

        # step7: execute normalization
        xhat = xmu * ivar

        # step8: Nor the two transformation steps
        gammax = gamma * xhat

        # step9
        out = gammax + beta

        # store running averages of mean and variance during training for use during testing
        if self.ema_mean is None or self.ema_var is None:
            self.ema_mean = mu
            self.ema_var = var
        else:
            self.ema_mean = self.momentum * self.ema_mean + (1 - self.momentum) * mu
            self.ema_var = self.momentum * self.ema_var + (1 - self.momentum) * var
        # store intermediate
        self.cache = (xhat, gamma, xmu, ivar, sqrtvar, var)

        return out

    def forward_pass(self, X):
        if len(X.shape) == 2:
            # input is a regular layer
            return self._forward_pass(X)
        elif len(X.shape) == 4:
            # input is a convolution layer
            N, C, H, W = X.shape
            x_flat = X.transpose(0, 2, 3, 1).reshape(-1, C)
            out_flat = self._forward_pass(x_flat)
            return out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            raise NotImplementedError('Unknown model with dimensions = {}'.format(len(X.shape)))

    def _backward_pass(self, delta):
        # unfold the variables stored in cache
        xhat, gamma, xmu, ivar, sqrtvar, var = self.cache

        # get the dimensions of the input/output
        N, D = delta.shape

        # step9
        dbeta = np.sum(delta, axis=0)
        dgammax = delta  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step7
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(var + self.eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        # Update gradient values
        self._params.update_grad('W', dgamma)
        self._params.update_grad('b', dbeta)

        return dx

    def backward_pass(self, X):
        if len(X.shape) == 2:
            # input is a regular layer
            return self._backward_pass(X)
        elif len(X.shape) == 4:
            # input is a convolution layer
            N, C, H, W = X.shape
            x_flat = X.transpose(0, 2, 3, 1).reshape(-1, C)
            out_flat = self._backward_pass(x_flat)
            return out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            raise NotImplementedError('Unknown model shape: {}'.format(X.shape))

    def shape(self, x_shape):
        return x_shape
