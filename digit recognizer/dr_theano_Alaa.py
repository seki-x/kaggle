# -*- coding: utf-8 -*-
"""
@author: Alaa Awad
Source: https://www.kaggle.com/alaaawad/digit-recognizer/tensor-theano/code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import pickle
import time
theano.config.exception_verbosity = 'high'

train = pd.read_csv("data\\train1002.csv")
test = pd.read_csv("data\\test502.csv")
trX = np.array(train.values[:][:, 1:], dtype=np.float32)/256
trY = np.array(train.values[:][:, 0], dtype=int)
teX = np.array(test.values[:], dtype=np.float32)/256
trY_onehot = np.zeros((trY.shape[0], 10), dtype=np.float32)
trY_onehot[np.arange(trY.shape[0]), trY] = 1


def make_submission_csv(predict, is_list=False):
    if is_list:
        df = pd.DataFrame({'ImageId': range(1, 28001), 'Label': predict})
        df.to_csv("submit.csv", index=False)
        return
    pred = []
    for i in range(28000):
        pred.append(predict(test.values[i]))
    df = pd.DataFrame({'ImageId': range(1, 28001), 'Label': pred})
    df.to_csv("submit.csv", index=False)


def rectify(Z):
    return T.maximum(Z, 0.)


def init_weights(shape):
    return theano.shared(np.random.randn(*shape)*0.01)


def get_updates(cost, params, lr=np.float32(0.05)):
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        updates.append([p, p - (g * lr)])
    return updates


def model(X, w_h, w_o):
    h = rectify(T.dot(X, w_h))
    return T.nnet.softmax(T.dot(h, w_o))

w_h = theano.shared(np.random.randn(784, 60).astype(np.float32)*0.01, name='w_h')
w_o = theano.shared(np.random.randn(60, 10).astype(np.float32)*0.01, name='w_o')
X = T.fmatrix(name='X')
labels = T.fmatrix(name='labels')
prediction = model(X, w_h, w_o)
cost = T.mean(T.nnet.categorical_crossentropy(prediction, labels))
updates = get_updates(cost, [w_h, w_o])
train_func = theano.function(
    inputs=[X, labels], outputs=cost, updates=updates,
    allow_input_downcast=True)
predict_func = theano.function(
    inputs=[X], outputs=prediction, allow_input_downcast=True)

costs = []
niters = 200
t = time.clock()
for i in range(niters):
    print("Iter: "+str(i))
    costt = train_func(trX, trY_onehot)
    print("Cost: "+str(costt))
    costs.append(float(costt))
    print("time ", (time.clock()-t))
    t = time.clock()

pickle.dump(costs, open("costs.p", 'wb'))
plt.scatter(np.arange(len(costs)), costs)
plt.savefig("cost.png")
plt.show()
make_submission_csv(np.argmax(predict_func(teX), axis=1), is_list=True)