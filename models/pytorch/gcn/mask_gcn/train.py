from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from types import SimpleNamespace

import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils import *
from scipy import sparse
from models import GCN_MASK
import scipy.io as scio
import pandas as pd
import pickle
import pdb



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../../data/multitask_dataset.pkl', help='Data path.')
parser.add_argument('--model', type=str, default='gcn_mask', help='Model string.')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping',type=int, default=10, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--seed', type=int, default=6, help='define the seed.')
parser.add_argument('--train_percentage', type=float, default=0.1 , help='define the percentage of training data.')
parser.add_argument('--fastgcn_setting', type=int, default=0, help='define the training setting for gcn or fastgcn setting')
parser.add_argument('--start_test', type=int, default=80, help='define from which epoch test')
parser.add_argument('--train_jump', type=int, default=0, help='define whether train jump, defaul train_jump=0')
parser.add_argument('--attack_dimension', type=int, default=0, help='define how many dimension of the node feature to attack')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

k_att = args.train_percentage
test_result_gather = []

# Load data
add_all, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.fastgcn_setting, 
                                                                                        args.dataset,
                                                                                        k_att, args.attack_dimension,
                                                                                        args.train_jump)
# Some preprocessing

features = preprocess_features(features,adj) ## type(features) is tuple

if args.model == 'gcn_mask':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_MASK
elif args.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, args.max_degree)
    num_supports = 1 + args.max_degree
    model_func = GCN
elif args.model == 'dense':
    support = [preprocess_adj(adj)]  # Not usedouts
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))

# Create model
model = model_func(add_all, input_dim=features[2][1], logging=True)

# Define model evaluation function
def evaluate(features, support, labels, mask):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask)
    outs_val = sess.run([model.loss, model.accuracy, model.mask], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

# Init variables

cost_val = []
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

train_gcnmask_gather = []
test_gcnmask_gather = []
val_gcnmask_gather = []

best_test_result = 0

# Train model
for epoch in range(args.epochs):

    t = time.time()

    feed_dict = construct_feed_dict(features, support, y_train, train_mask)

    feed_dict.update({args.dropout: args.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.mask], feed_dict=feed_dict)
    ## the last layer of nn output is model.outputs
    cost, acc, duration, val_gcnmask = evaluate(features, support, y_val, val_mask)

    cost_val.append(cost)  ##transpose to numpy and reshape, then write to txt
    train_loss.append(outs[1])
    train_accuracy.append(outs[2])
    val_loss.append(cost)
    val_accuracy.append(acc)

    
    train_gcnmask_gather.append(outs[4])
    val_gcnmask_gather.append(val_gcnmask)
    # set the preserved values
    np.set_printoptions(precision=3)


    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
    if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
        print("aaa=========",epoch)
        print("Early stopping...")
        break

    if epoch > args.start_test:

        test_cost, test_acc, test_duration, test_gcnmask = evaluate(features, support, y_test, test_mask)

        if test_acc > best_test_result:
            best_test_result = test_acc

        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        
feed_dict_test = construct_feed_dict(features, support, y_test, test_mask)
test_cost, test_acc, test_duration, test_gcnmask = evaluate(features, support, y_test, test_mask)
test_gcnmask_gather = test_gcnmask

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_result_gather.append([k_att, best_test_result])
