import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, BCEWithLogitsLoss, MSELoss


class CEWithREF(Module):
    def __init__(self, w_ref, mode='train'):
        super(CEWithREF, self).__init__()
        self.ce = CrossEntropyLoss()
        self.mse = MSELoss(reduction='none')
        self.w_ref = w_ref
        self.mode = mode

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.mode == 'train':
            loss = 0
            for pred in input:
                loss += self.ce(pred, target)
                loss += self.w_ref * self.mse(torch.log_softmax(pred[1:, :], dim=1), torch.log_softmax(pred.detach()[:-1, :], dim=1)).clamp(0, 16).mean()
        else:
            pred = input[-1]
            loss = self.ce(pred, target) + self.w_ref * self.mse(torch.log_softmax(pred[1:, :], dim=1), torch.log_softmax(pred.detach()[:-1, :], dim=1)).clamp(0, 16).mean()

        return loss


_LOSSES = {
          'ce':              CrossEntropyLoss,
          'bce_logit':       BCEWithLogitsLoss,
          'mse':             MSELoss,
          'ce_ref':          CEWithREF
          }


def get_loss_func(cfg, mode='train'):
    """
    Get the loss function corresponding to "loss_name"
    """

    loss_name = cfg['loss_name']
    if loss_name not in _LOSSES:
        raise ValueError(f'Loss {loss_name} is not implemented in models/losses.py')

    if cfg['use_ref']:
        return _LOSSES[loss_name](cfg['w_ref'], mode)

    return _LOSSES[loss_name]()
