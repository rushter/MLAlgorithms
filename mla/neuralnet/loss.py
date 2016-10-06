from ..metrics import mse, logloss, mae, hinge, binary_crossentropy

categorical_crossentropy = logloss


def get_loss(name):
    """Return loss function by name"""

    try:
        return globals()[name]
    except:
        raise ValueError('Invalid metric function.')
