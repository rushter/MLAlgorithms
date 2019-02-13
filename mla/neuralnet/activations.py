import autograd.numpy as np

"""
References:
https://en.wikipedia.org/wiki/Activation_function
"""


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    # Avoid numerical overflow by removing max
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def linear(z):
    return z


def softplus(z):
    """Smooth relu."""
    # Avoid numerical overflow, see:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.logaddexp.html
    return np.logaddexp(0.0, z)


def softsign(z):
    return z / (1 + np.abs(z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


def leakyrelu(z, a=0.01):
    return np.maximum(z * a, z)


def get_activation(name):
    """Return activation function by name"""
    try:
        return globals()[name]
    except:
        raise ValueError('Invalid activation function.')
