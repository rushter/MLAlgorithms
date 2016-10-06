import numpy as np


def normal(shape, scale=0.5):
    return np.random.normal(size=shape, scale=scale)


def uniform(shape, scale=0.5):
    return np.random.uniform(size=shape, low=-scale, high=scale)


def zero(shape, **kwargs):
    return np.zeros(shape)


def one(shape, **kwargs):
    return np.ones(shape)


def orthogonal(shape, scale=0.5):
    flat_shape = (shape[0], np.prod(shape[1:]))
    array = np.random.normal(size=flat_shape)
    u, _, v = np.linalg.svd(array, full_matrices=False)
    array = u if u.shape == flat_shape else v
    return np.reshape(array * scale, shape)


def _glorot_fan(shape):
    assert len(shape) >= 2

    if len(shape) == 4:
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in, fan_out = shape[:2]
    return float(fan_in), float(fan_out)


def glorot_normal(shape, **kwargs):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)


def glorot_uniform(shape, **kwargs):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)


def he_normal(shape, **kwargs):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s)


def he_uniform(shape, **kwargs):
    fan_in, fan_out = _glorot_fan(shape)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s)


def get_initializer(name):
    """Return initialization function by name"""
    try:
        return globals()[name]
    except:
        raise ValueError('Invalid initialization function.')
