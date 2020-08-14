import logging
import torch

_logger = logging.getLogger(__name__)


def heteroscedastic_loss(output, gt_output, reduction):
    """
    Args:
        output: NN output values, tensor of shape 2, batch_size, n_channels.
        where the first dimension contains the mean values and the second
        dimension contains the log_var
        gt_output: groundtruth values. tensor of shape batch_size, n_channels
        reduction: if mean, the loss is averaged across the third dimension, 
        if summ the loss is summed across the third dimension, if None any 
        aggregation is performed

    Returns:
        tensor of size n_channels if reduction is None or tensor of size 0
        if reduction is mean or sum

    """
    precision = torch.exp(-output[1])
    batch_size = output[0].shape[0]
    loss = torch.sum(precision * (gt_output - output[0]) ** 2. 
                     + output[1], 0)/batch_size
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction is None:
        return loss
    else:
        _logger.error('Aggregation can only be None, mean or sum.')
    

class HeteroscedasticLoss(torch.nn.Module):
    """
    Heteroscedastic loss
    """
    def __init__(self, reduction='mean'):
        super(HeteroscedasticLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        return heteroscedastic_loss(output, target, reduction=self.reduction)
    