# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:32:34 2016

@author: xu labo
"""

import mxnet as mx
cpu_tensor = mx.nd.zeros((10,), ctx=mx.cpu())
#gpu_tensor = mx.nd.zeros((10,), ctx=mx.gpu(0))
ctx = mx.cpu() # which context to put memory on.
a = mx.nd.ones((10, ), ctx=ctx)
b = mx.nd.ones((10, ), ctx=ctx)
c = (a + b) / 10.
d = b + 1

import mxnet as mx
a = mx.sym.Variable("A") # represent a placeholder. These can be inputs, weights, or anything else.
b = mx.sym.Variable("B")
c = (a + b) / 10
d = c + 1


d.list_arguments()
# ['A', 'B']

d.list_outputs()
# ['_plusscalar0_output'] This is the default name from adding to scalar.


# define input shapes
inp_shapes = {'A':(10,), 'B':(10,)}
arg_shapes, out_shapes, aux_shapes = d.infer_shape(**inp_shapes)

arg_shapes # the shapes of all the inputs to the graph. Order matches d.list_arguments()
# [(10, ), (10, )]

out_shapes # the shapes of all outputs. Order matches d.list_outputs()
# [(10, )]

aux_shapes # the shapes of auxiliary variables. 
# These are variables that are not trainable such as batch normalization 
# population statistics. For now, they are save to ignore.

# []

input_arguments = {}
input_arguments['A'] = mx.nd.ones((10, ), ctx=mx.cpu())
input_arguments['B'] = mx.nd.ones((10, ), ctx=mx.cpu())
executor = d.bind(ctx=mx.cpu(),
                  args=input_arguments, 
                  # this can be a list or a dictionary mapping names of inputs to NDArray
                  grad_req='null') # don't request gradients


import numpy as np
# The executor
executor.arg_dict
# {'A': NDArray, 'B': NDArray}

executor.arg_dict['A'][:] = np.random.rand(10,) 
# Note the [:]. This sets the contents of the array instead of setting the array 
# to a new value instead of overwriting the variable.
executor.arg_dict['B'][:] = np.random.rand(10,)
executor.forward()
executor.outputs
# [NDArray]
output_value = executor.outputs[0].asnumpy()


# allocate space for inputs
input_arguments = {}
input_arguments['A'] = mx.nd.ones((10, ), ctx=mx.cpu())
input_arguments['B'] = mx.nd.ones((10, ), ctx=mx.cpu())
# allocate space for gradients
grad_arguments = {}
grad_arguments['A'] = mx.nd.ones((10, ), ctx=mx.cpu())
grad_arguments['B'] = mx.nd.ones((10, ), ctx=mx.cpu())

executor = d.bind(ctx=mx.cpu(),
                  args=input_arguments, # this can be a list or a dictionary mapping names of inputs to NDArray
                  args_grad=grad_arguments, # this can be a list or a dictionary mapping names of inputs to NDArray
                  grad_req='write') # instead of null, tell the executor to write gradients. This replaces the contents of grad_arguments with the gradients computed.

executor.arg_dict['A'][:] = np.random.rand(10,)
executor.arg_dict['B'][:] = np.random.rand(10,)

executor.forward()
# in this particular example, the output symbol is not a scalar or loss symbol.
# Thus taking its gradient is not possible.
# What is commonly done instead is to feed in the gradient from a future computation.
# this is essentially how backpropagation works.
out_grad = mx.nd.ones((10,), ctx=mx.cpu())
executor.backward([out_grad]) # because the graph only has one output, only one output grad is needed.

executor.grad_arrays
# [NDarray, NDArray]








