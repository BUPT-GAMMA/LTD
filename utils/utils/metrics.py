import numpy as np
import torch


def accuracy(output, labels, details=False, hop_idx=None,w=False, te = [],g=[],idx=[],student=[],real=[],tea=[],idx_tes=[],ft=[]):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    result = correct.sum()
    if details:
        hop_num = np.bincount(hop_idx, minlength=7)
        true_idx = np.array((correct > 0).nonzero().squeeze(dim=1).cpu())
        true_hop = np.bincount(hop_idx[true_idx], minlength=7)/hop_num
        return result / len(labels), true_hop
    if w:
        true_idx = []
        tea_pre = []
        output = torch.exp(output)
    return result / len(labels)


def eucli_dist(output, target):
    return torch.sqrt(torch.sum(torch.pow((output - target), 2)))


def my_loss(output, target, mode=0):
    if mode == 0:
        return eucli_dist(torch.exp(output), target)
    elif mode == 1:
        # Distilling the Knowledge in a Neural Network
        return torch.nn.BCELoss()(torch.exp(output), target)
    elif mode == 2:
        # Exploring Knowledge Distillation of Deep Neural Networks for Efficient Hardware Solutions
        return torch.nn.KLDivLoss()(output, target)