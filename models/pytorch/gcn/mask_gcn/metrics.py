#import tensorflow as tf
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax_cross_entropy(preds, labels, mask):
    #pdb.set_trace()
    #print("##### masked_softmax cross entropy ####")
    """Softmax cross-entropy loss with masking."""
    preds = F.softmax(preds)
    loss = -torch.sum(labels * torch.log(preds), 1)
    loss = torch.unsqueeze(loss, 1)
    mask = torch.unsqueeze(mask, 1)
    mask /= torch.mean(mask)
    loss = torch.mul(loss, mask)
    return torch.mean(loss)


def masked_accuracy(preds, labels, mask):
    #pdb.set_trace()
    #print("##### masked_accuracy ####")
    """Accuracy with masking."""
    correct_prediction = torch.equal(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all =torch.unsqueeze(correct_prediction, 1)
    mask = torch.unsqueeze(mask, 1)
    mask /= torch.mean(mask)
    accuracy_all = torch.mul(accuracy_all, mask)
    return torch.mean(accuracy_all)
