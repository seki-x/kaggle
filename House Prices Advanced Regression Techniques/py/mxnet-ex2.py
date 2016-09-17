# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:10:16 2016

@author: john
"""

import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # get a logger to accuracies are printed

data = mx.sym.Variable("data") # input features, when using FeedForward this must be called data
label = mx.sym.Variable("softmax_label") # use this name aswell when using FeedForward

# When using Forward its best to have mxnet create its own variables.
# The names of them are used for initializations.
l1 = mx.sym.FullyConnected(data=data, num_hidden=128, name="layer1")
a1 = mx.sym.Activation(data=l1, act_type="relu", name="act1")
l2 = mx.sym.FullyConnected(data=a1, num_hidden=10, name="layer2")

cost_classification = mx.sym.SoftmaxOutput(data=l2, label=label)

from skdata.mnist.views import OfficialVectorClassification

data = OfficialVectorClassification()
trIdx = data.sel_idxs[:]
teIdx = data.val_idxs[:]

model = mx.model.FeedForward(symbol=cost_classification,
                             num_epoch=10,
                             ctx=mx.cpu(),
                             learning_rate=0.001)

model.fit(X=data.all_vectors[trIdx],
          y=data.all_labels[trIdx],
          eval_data=(data.all_vectors[teIdx], data.all_labels[teIdx]),
          eval_metric="acc",
          logger=logger)