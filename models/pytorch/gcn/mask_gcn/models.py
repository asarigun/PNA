
from layers import *
from metrics import masked_softmax_cross_entropy
from metrics import masked_accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.args = {}

        self.layers = []
        self.activations = []
        self.add_all = []
        self.mask = []

        self.inputs = None
        self.outputs = None
        self.predict = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1] ## activations[-1] get the last layers output


        # Build metrics
        self._loss()
        self._accuracy()
        self._mask()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _mask(self):
        pass
    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

class MLP(Model):
    def __init__(self, args, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = args.features
        self.input_dim = input_dim
        self.output_dim = args.labels.get_shape().as_list()[1]
        self.args = args

        self.optimizer = optim.Adam(model.parameters(), learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += args.weight_decay * torch.nn.MSELoss(var, size_average=None, reduce=None, reduction='mean')

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.args.labels,
                                                  self.args.labels_mask)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.args.labels,
                                        self.args.labels_mask)

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=args.hidden1,
                                 args=self.args,
                                 act=torch.nn.ReLU(),
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=args.hidden1,
                                 output_dim=self.output_dim,
                                 args=self.args,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return torch.nn.Softmax(self.outputs)


class GCN_MASK(Model):

    def __init__(self, add_all, args, input_dim, **kwargs):
        super(GCN_MASK, self).__init__(**kwargs)

        self.inputs = args.features
        self.input_dim = input_dim
        self.add_all = add_all
        self.output_dim = args.labels.get_shape().as_list()[1]
        self.args = args
        self.optimizer = optim.Adam(model.parameters(), learning_rate=args.learning_rate, weight_decay=args.weight_decay)      
        self.build()
        
   
    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += args.weight_decay * torch.nn.MSELoss(var, size_average=None, reduce=None, reduction='mean')

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.args.labels,
                                                  self.args.labels_mask)

    def _accuracy(self):

        self.accuracy = masked_accuracy(self.outputs, self.args.labels,
                                        self.args.labels_mask)

    def _build(self):
       
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=args.hidden1,
                                            args=self.args,
                                            act=torch.nn.ReLU(),
                                            dropout=True,
                                            sparse_inputs= True,
                                            logging=self.logging))                                  
                                      
        self.layers.append( gcnmask(add_all = self.add_all,
                         input_dim=args.hidden1,
                         output_dim=self.output_dim,
                         args=self.args,
                         #act=lambda x: x,
                         act = torch.nn.ReLU(),
                         dropout=True,
                         logging=self.logging))
        
    
    def _mask(self):
        return self.mask    

    def predict(self):
        return torch.nn.Softmax(self.outputs)
