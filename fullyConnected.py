# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:22:59 2016

@author: Peyman Beyrnavand
"""
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, Pool2DLayer
from lasagne.layers import get_output, get_all_params, get_all_param_values
from lasagne.layers import Conv2DLayer, set_all_param_values, batch_norm
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.regularization import regularize_network_params, l1, l2
from lasagne.updates import rmsprop
from lasagne.init import GlorotUniform
import theano.tensor as T
from theano import function as Tfunc
from theano import shared
from theano import config
import numpy as np
import time
import cPickle as pickle


def shared_dataset(data):
    # load data to shared variable(implicitly GPU)
    X_train = shared(data['X_train'].astype(config.floatX),
                     borrow=True)
    y_train = T.cast(shared(data['y_train'].astype(config.floatX),
                            borrow=True),
                     'int32')
    X_val = shared(data['X_val'].astype(config.floatX),
                   borrow=True)
    y_val = T.cast(shared(data['y_val'].astype(config.floatX),
                          borrow=True),
                   'int32')
    X_test = shared(data['X_test'].astype(config.floatX),
                    borrow=True)
    y_test = T.cast(shared(data['y_test'].astype(config.floatX),
                           borrow=True),
                    'int32')
    return (X_train, y_train, X_val, y_val, X_test, y_test)


def reinitiate_set_params(network,
                          weights=None):
        # reset weights of trained network torandom or user defined value
        # useful in case of big networks and cross validation
        # instead of the long time of recompiling you can just
        # re-init the network weights
        if not weights:
            old = get_all_param_values(network)
            weights = []
            for layer in old:
                shape = layer.shape
                if len(shape) < 2:
                    shape = (shape[0], 1)
                W = GlorotUniform()(shape)
                if W.shape != layer.shape:
                    W = np.squeeze(W, axis=1)
                weights.append(W)
        set_all_param_values(network, weights)
        return network


def build_fc_model(in_shape,
                   num_hidden,
                   num_out,
                   dropout=None,
                   nlin_func=rectify,
                   in_var=None):
    """
    Build a fully connected network

    :in_shape: tuple, shape of input, (#samples, #channels, width, heights)
    :num_hidden: list of int, list of number of units for each hidden layer
    :dropout: list of float, for each hidden layer what prob to drop nodes
    :num_out: int, number of output classes for prediction
    :nlin_func: lasagne.nonlinearities, function to apply in hidden layers
    :in_var: theano.T, symolic tensor variable to represent the input
    """

    # input layer
    net = InputLayer(input_var=in_var,
                     shape=in_shape)
    # hidden layers
    for i in xrange(len(num_hidden)):
        net = DenseLayer(incoming=net,
                         num_units=num_hidden[i],
                         nonlinearity=nlin_func)
        if dropout:
            net = DropoutLayer(net, p=dropout[i])
    # output layer
    net = DenseLayer(incoming=net,
                     num_units=num_out,
                     nonlinearity=softmax)
    return net


def build_fc_model_batchnorm(in_shape,
                             num_hidden,
                             num_out,
                             nlin_func=rectify,
                             in_var=None):
    """
    Build a fully connected network with batchnormalisation

    :in_shape: tuple, shape of input, (#samples, #channels, width, heights)
    :num_hidden: list of int, list of number of units for each hidden layer
    :dropout: list of float, for each hidden layer what prob to drop nodes
    :num_out: int, number of output classes for prediction
    :nlin_func: lasagne.nonlinearities, function to apply in hidden layers
    :in_var: theano.T, symolic tensor variable to represent the input
    """

    # input layer
    net = InputLayer(input_var=in_var,
                     shape=in_shape)

    # hidden layers
    for i in xrange(len(num_hidden)):
        net = batch_norm(DenseLayer(incoming=net,
                                    num_units=num_hidden[i],
                                    nonlinearity=nlin_func))

    # output layer
    net = DenseLayer(incoming=net,
                     num_units=num_out,
                     nonlinearity=softmax)
    return net


def build_CNN(in_shape,
              num_hidden,
              num_filter,
              num_out,
              fil_size,
              nlin_func=rectify,
              in_var=None):
    # build a CNN
    net = InputLayer(input_var=in_var,
                     shape=in_shape)
    net = Conv2DLayer(net,
                      num_filters=num_filter,
                      filter_size=fil_size,
                      stride=1,
                      pad=1,
                      nonlinearity=nlin_func,
                      flip_filters=False)

    net = Pool2DLayer(net,
                      pool_size=2,
                      stride=2)

    net = DenseLayer(incoming=net,
                     num_units=num_hidden[0],
                     nonlinearity=nlin_func)

    net = DenseLayer(incoming=net,
                     num_units=num_hidden[1],
                     nonlinearity=nlin_func)

    net = DenseLayer(incoming=net,
                     num_units=num_out,
                     nonlinearity=softmax)
    return net


def build_CNN_batchnorm(in_shape,
                        num_hidden,
                        num_filter,
                        num_out,
                        fil_size,
                        nlin_func=rectify,
                        in_var=None):
    # build a CNN
    net = InputLayer(input_var=in_var,
                     shape=in_shape)
    net = batch_norm(Conv2DLayer(net,
                                 num_filters=num_filter,
                                 filter_size=fil_size,
                                 stride=1,
                                 pad=1,
                                 nonlinearity=nlin_func,
                                 flip_filters=False))

    net = Pool2DLayer(net,
                      pool_size=2,
                      stride=2)

    net = batch_norm(DenseLayer(incoming=net,
                                num_units=num_hidden,
                                nonlinearity=nlin_func))

    net = batch_norm(DenseLayer(incoming=net,
                                num_units=num_out,
                                nonlinearity=softmax))
    return net


def build_2xCNN_batchnorm(in_shape,
                          num_hidden,
                          num_filter,
                          num_out,
                          fil_size,
                          nlin_func=rectify,
                          in_var=None):
    # build a CNN
    net = InputLayer(input_var=in_var,
                     shape=in_shape)
    net = batch_norm(Conv2DLayer(net,
                                 num_filters=num_filter[0],
                                 filter_size=fil_size[0],
                                 stride=1,
                                 pad=1,
                                 nonlinearity=nlin_func,
                                 flip_filters=False))

    net = Pool2DLayer(net,
                      pool_size=2,
                      stride=2)

    net = batch_norm(Conv2DLayer(net,
                                 num_filters=num_filter[1],
                                 filter_size=fil_size[1],
                                 stride=1,
                                 pad=1,
                                 nonlinearity=nlin_func,
                                 flip_filters=False))

    net = Pool2DLayer(net,
                      pool_size=2,
                      stride=2)

    net = batch_norm(DenseLayer(incoming=net,
                                num_units=num_hidden,
                                nonlinearity=nlin_func))

    net = batch_norm(DenseLayer(incoming=net,
                                num_units=num_out,
                                nonlinearity=softmax))
    return net


def build_CNN_nopool(in_shape,
                     num_filter,
                     fil_size,
                     strides,
                     num_out,
                     nlin_func=rectify,
                     in_var=None):

    # build a CNN
    net = InputLayer(input_var=in_var,
                     shape=in_shape)

    for i in xrange(len(fil_size)):
        net = batch_norm(Conv2DLayer(net,
                                     num_filters=num_filter[i],
                                     filter_size=fil_size[i],
                                     stride=strides[i],
                                     pad=1,
                                     nonlinearity=nlin_func,
                                     flip_filters=False))

    net = DenseLayer(incoming=net,
                     num_units=num_out,
                     nonlinearity=softmax)

    return net


def update_functions(net, data,
                     in_var, target,
                     l1_reg=0,
                     l2_reg=0,
                     batch_size=16,
                     lear_rate=1e-3):

    """
    build update expression and prediction of the network
    :net: lasagne.layers, final layer of the network
    :in_var: theano.T, variable representing the input of the network
    :target: theano.T, variable representing the actual output
    :l1_reg: float, weight for l1 reguarization in [0,1]
    :l2_reg: floatweight for l2 regularization in [0,1]
     this should hold: 1 - l1_reg + l2_reg <= 1
    :lear_rate: float, learning rate of rmsprop algorithm
    """

    # get network predictions
    prediction = get_output(net)

    # compute cross entropy loss
    train_loss = T.mean(categorical_crossentropy(predictions=prediction,
                                                 targets=target))
    train_loss *= (1 - l1_reg - l2_reg)
    # compute regularization loss
    if l1_reg:
        loss_regl1 = regularize_network_params(net, l1)
        train_loss += l1_reg * loss_regl1

    if l2_reg:
        loss_regl2 = regularize_network_params(net, l2)
        train_loss += l2_reg * loss_regl2

    # get parameters for gradient calculation
    params = get_all_params(net, trainable=True)

    # compute updates
    update = rmsprop(train_loss, params, learning_rate=1e-3)

    # compute validation loss
    test_prediction = get_output(net,
                                 deterministic=True  # necessary for Dropout
                                 )
    test_loss = T.mean(categorical_crossentropy(predictions=test_prediction,
                                                targets=target))

    # calculate the predicted class
    y_pred = T.argmax(test_prediction, axis=1)
    # calculate the proportion of errors that has been made
    errors = T.mean(T.neq(y_pred, target))

    X_train, y_train, X_val, y_val, X_test, y_test = data
    #################################################
    # build training, validation and test functions

    # index for mini-batch slicing
    index = T.lscalar()

    # training function
    size_train = X_train.get_value().shape[0]

    train_fn = Tfunc(inputs=[index],
                     outputs=[train_loss],
                     updates=update,
                     givens={in_var: X_train[index * batch_size: T.minimum((index + 1) * batch_size, size_train)],
                             target: y_train[index * batch_size: T.minimum((index + 1) * batch_size, size_train)]})
    # validation function
    val_fn = Tfunc(inputs=[],
                   outputs=[test_loss, errors],
                   givens={in_var: X_val, target: y_val})

    # test function
    test_fn = Tfunc(inputs=[],
                    outputs=[y_pred, errors],
                    givens={in_var: X_test, target: y_test})
    return train_fn, val_fn, test_fn


def early_stop(net, data,
               X, y,
               batch_size=16,
               l1_reg=0,
               l2_reg=0,
               lear_rate=1e-3,
               saveHistory=False,
               name='trainHistory',
               iteration=10,
               printFreq=60):
    """
    :net: lasagne.layers, final layer of the network
    :data:tuple(theano.shared), (X_train, y_train, X_val, y_val, X_test,y_test)
    :X: theano.tesnor, symbolic variable pointing to input
    :y: theano.tensor, symbolic variable pointing to output
    :batch_size: int, training mini batch size
    :l1_reg: float in [0,1], proprtion of l1 regularization
    :l2_reg: float in [0,1], proportion of l2 reguarization
    :lear_rate: float [0,1], learning rate for rmsprop
    :name: string, name of the file to save the training variables
    """
    # build update functions
    train_fn, val_fn, test_fn = update_functions(net, data=data,
                                                 in_var=X, target=y,
                                                 l1_reg=l1_reg, l2_reg=l2_reg,
                                                 lear_rate=lear_rate,
                                                 batch_size=batch_size)

    X_train, _, X_val, _, _, _ = data
    n_train_batches = X_train.get_value(borrow=True).shape[0] // batch_size + 1
    print 'shape training', X_train.get_value(borrow=True).shape, '\n'
    print 'shape validation', X_val.get_value(borrow=True).shape, '\n'
    del X_train, X_val

    n_iter = iteration
    improvement_threshold = 0.998
    patience = 40000
    patience_increase = .3
    validation_frequency = min(n_train_batches, patience // 10)
    print 'validation_frequency', validation_frequency

    train_loss_history_temp = []
    best_val_loss_ = np.inf
    epoch = 0
    done_looping = False
    train_loss_history_ = []
    val_loss_history_ = []
    val_error_history_ = []

    print 'start training'
    start_time = time.time()

    while (epoch < n_iter) and (not done_looping):
        epoch += 1

        # go over mini-batches for a full epoch
        for minibatch_index in range(n_train_batches):

            # update network for one mini-batch
            minibatch_loss = train_fn(minibatch_index)

            # store training loss of mini-batches till the next validation step
            train_loss_history_temp.append(minibatch_loss)

            # number of mini-batches checked
            num_minibatch_checked = (epoch - 1) * n_train_batches + minibatch_index

            # if validation interval reached
            if (num_minibatch_checked + 1) % validation_frequency == 0:

                # compute validation loss
                current_val_loss, current_val_error = val_fn()

                # store training and validation history
                train_loss_history_.append(np.mean(train_loss_history_temp))
                val_loss_history_.append(current_val_loss)
                val_error_history_.append(current_val_error)
                train_loss_history_temp = []

                # is it the best validation loss so far?
                if current_val_loss < best_val_loss_:

                    # increase patience if improvement is significant
                    if (current_val_loss < best_val_loss_ * improvement_threshold):
                        patience = max(patience, num_minibatch_checked * patience_increase)

                    # save the-so-far-best validation RMSE and epoch and model-params
                    best_val_loss_ = current_val_loss
                    best_network_params = get_all_param_values(net)

                    # save the best model as pickle file
                    if saveHistory:
                        pickle.dump([best_network_params, train_loss_history_,
                                     val_loss_history_, val_error_history_, net],
                                     open(name+".p", "wb"))
                            

            # check if patience exceeded and set the training loop to stop
            if (patience <= num_minibatch_checked):
                print 'patience reached \n'
                # reset the network weights to the best params saved
                print 'resetting network params to that of the best seen \n'
                reinitiate_set_params(network=net,
                                      weights=best_network_params)
                # done optimising, break the optimisation loop
                done_looping = True
                break

        if (epoch % printFreq) == 1:
            print (('epoch %i, val_loss %f, train_loss %f, error %f, best error %f ,in %f secs \n') %
                   (epoch, current_val_loss,train_loss_history_[-1], current_val_error, min(val_error_history_), time.time() - start_time))
            start_time = time.time()

    return net, test_fn, train_loss_history_, val_loss_history_, val_error_history_
