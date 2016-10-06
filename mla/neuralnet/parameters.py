import numpy as np

from mla.neuralnet.initializations import get_initializer


class Parameters(object):
    def __init__(self, init='glorot_uniform', scale=0.5, bias=1.0, regularizers=None, constraints=None):
        """A container for layer's parameters.

        Parameters
        ----------
        init : str, default 'glorot_uniform'.
            The name of the weight initialization function.
        scale : float, default 0.5
        bias : float, default 1.0
            Initial values for bias.
        regularizers : dict
            Weight regularizers.
            {'W' : L2()}
        constraints : dict
            Weight constraints. {'b' : MaxNorm()}
        """
        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = constraints

        if regularizers is None:
            self.regularizers = {}
        else:
            self.regularizers = regularizers

        self.initial_bias = bias
        self.scale = scale
        self.init = get_initializer(init)

        self._params = {}
        self._grads = {}

    def setup_weights(self, W_shape, b_shape=None):
        if 'W' not in self._params:
            self._params['W'] = self.init(shape=W_shape, scale=self.scale)
            if b_shape is None:
                self._params['b'] = np.full(W_shape[1], self.initial_bias)
            else:
                self._params['b'] = np.full(b_shape, self.initial_bias)
        self.init_grad()

    def init_grad(self):
        """Init gradient arrays corresponding to each weight array."""
        for key in self._params.keys():
            if key not in self._grads:
                self._grads[key] = np.zeros_like(self._params[key])

    def step(self, name, step):
        """Increase specific weight by amount of the step parameter."""
        self._params[name] += step

        if name in self.constraints:
            self._params[name] = self.constraints[name].clip(self._params[name])

    def update_grad(self, name, value):
        self._grads[name] = value

        if name in self.regularizers:
            self._grads[name] += self.regularizers[name](self._params[name])

    @property
    def n_params(self):
        """Count the number of parameters in this layer."""
        return sum([np.prod(self._params[x].shape) for x in self._params.keys()])

    def keys(self):
        return self._params.keys()

    @property
    def grad(self):
        return self._grads

    # Allow access to the fields using dict syntax, e.g. parameters['W']
    def __getitem__(self, item):
        if item in self._params:
            return self._params[item]
        else:
            raise ValueError

    def __setitem__(self, key, value):
        self._params[key] = value
