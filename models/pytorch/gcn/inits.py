#import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = torch.randint(low=-scale, high=scale, size=shape, generator=None, out=None, dtype=torch.float32, layout=torch.strided, device=None, requires_grad=True)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.randint(low=-init_range, high=init_range, size=shape, generator=None, out=None, dtype=torch.float32, layout=torch.strided, device=None, requires_grad=True)

def zeros(shape, name=None):
    """All zeros."""
    initial = torch.zeros(size=shape, dtype=torch.float32, requires_grad=True)
    
def ones(shape, name=None):
    """All ones."""
    initial = torch.zeros(size=shape, dtype=torch.float32, requires_grad=True)
