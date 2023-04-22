import torch.nn as nn


_LOSSES = {
          'cross_entropy':    nn.CrossEntropyLoss,
          'bce':              nn.BCELoss,
          'bce_logit':        nn.BCEWithLogitsLoss,
          'mse':              nn.MSELoss
          }


def get_loss_func(loss_name):
    """
    Get the loss function corresponding to "loss_name"
    """

    if loss_name not in _LOSSES:
        raise ValueError(f'Loss {loss_name} is not implemented in models/losses.py')

    return _LOSSES[loss_name]()
