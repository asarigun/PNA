from inits import *
import pdb
import argparse, json

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm


from torch.utils.tensorboard import SummaryWriter
import numpy as np



_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x):
        mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/self.kprob)
        return torch.sparse.FloatTensor(rc, val)



def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = spmm(x, y)
    else:
        res = torch.matmul(x, y)
    return res


def mask(mask):
    return mask


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        self.mask = []
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
    # def _mask (mask):
    #     return self.mask
    def _call(self, inputs):
        return inputs


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False, act=nn.ReLU(), bias=False, featureless=False, **kwargs): 
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = args.dropout 
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.device = device
        self.num_features_nonzero = args.num_features_nonzero

        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim), device=device))
        if bias:
            self.b = nn.Parameter(torch.zeros(output_dim, device=device))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()


        if self.logging:
            self._log_vars()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def _call(self, inputs):

        x = inputs

        # # dropout
        if self.sparse_inputs:
            x = SparseDropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = nn.Dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.W, sparse=self.sparse_inputs)
        
        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, dropout=0., 
                 sparse_inputs=False, act=nn.ReLU(), bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = args.dropout #placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        

        self.support = args.support #placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.device = device

        self.num_features_nonzero = args.num_features_nonzero


        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim), device=device))
        if bias:
            self.b = nn.Parameter(torch.zeros(output_dim, device=device))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()


        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = SparseDropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = nn.Dropout(x, 1-self.dropout)

        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.W, sparse=self.sparse_inputs)
            else:
                pre_sup = self.W
            
            # supports.append(pre_sup)
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)

        output = torch.add(supports)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class gcnmask(Layer):#for sum aggregator
    """Graph convolution layer."""
    def __init__(self, add_all, input_dim, output_dim, dropout=0., sparse_inputs=False, act=nn.ReLU(), bias=False, 
                 featureless=False, **kwargs):
        super(gcnmask, self).__init__(**kwargs)

        if dropout:
            self.dropout = args.dropout 
        else:
            self.dropout = 0.

        self.act = act
        
        self.add_all = add_all
        self.support = args.support 
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.mask = []

       
        self.device = device

        self.num_features_nonzero = args.num_features_nonzero

        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim), device=device))
        if bias:
            self.b = nn.Parameter(torch.zeros(output_dim, device=device))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()
 
        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        x = inputs

        # dropout
        if self.sparse_inputs:
            x = SparseDropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = nn.Dropout(x, 1-self.dropout)        

        x_new = []
        #mask_gather = []

        for i in range(len(self.add_all)):

            aa = torch.gather(x,[i])

            aa_tile = torch.tile(aa, [len(self.add_all[i]), 1]) # expand central
            #bb_nei = tf.expand_dims(x[self.add_all[i]],0)
            bb_nei = torch.gather(x,self.add_all[i])
            cen_nei = torch.cat([aa_tile, bb_nei],1)
          
            mask0 = dot(cen_nei, self.W, sparse = self.sparse_inputs)
            mask0 = nn.Sigmoid(mask0)
            mask = nn.Dropout(mask0, 1-self.dropout)
            
            self.mask.append(mask)


            new_cen_nei = aa + torch.sum(mask * bb_nei, 0, keepdims=True)
            x_new.append(new_cen_nei)
       
        x_new = torch.squeeze(x_new)    
        pre_sup = dot(x_new, self.W, sparse=self.sparse_inputs)

        return self.act(pre_sup)
