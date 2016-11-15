import numpy as np


def check_data(a, b):
    if isinstance(a, list) or isinstance(a, range):
        a = np.array(a)

    if isinstance(b, list) or isinstance(b, range):
        b = np.array(b)

    if type(a) != type(b):
        raise ValueError('Type mismatch: %s and %s' % (type(a), type(b)))

    if a.size != b.size:
        raise ValueError('Arrays must be equal in length.')
    return a, b


def validate_input(function):
    def wrapper(a, b):
        a, b = check_data(a, b)
        return function(a, b)

    return wrapper
