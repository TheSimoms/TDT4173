import os
import sys
import six
import csv
import time

import sklearn
import sklearn.utils
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy.io as sio

import Image

import theano
import theano.tensor as T
import lasagne


def load_image(directory_name, file_name):
    letter = directory_name.split('/')[-1]
    correct_label = ord(letter) - 97

    image = Image.open('%s/%s' % (directory_name, file_name))

    return np.array(image, dtype="float"), correct_label


def load_data():
    data = []
    labels = []

    for root, directory_names, files in os.walk('data', topdown=False):
        for directory_name in directory_names:
            path = '%s/%s' % (root, directory_name)

            for file_name in os.listdir(path):
                image = load_image(path, file_name)

                data.append(image[0])
                labels.append(image[1])

    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, random_state=1)

    return training_data, training_labels, testing_data, testing_labels


def save_result(prediction):
    label_list = \
        '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    result_csv = csv.writer(open('./data/result.csv', 'w'))
    result_csv.writerow(['ID', 'Class'])
    for idx, c in enumerate(prediction):
        result_csv.writerow([str(idx + 6284), label_list[c]])


def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 32 * 32 * 16

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 16 * 16 * 32

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 8 * 8 * 64

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 4 * 4 * 128

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 2 * 2 * 256

    network = lasagne.layers.GlobalPoolLayer(network)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=62,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def build_cnn_large(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 32 * 32 * 128

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 16 * 16 * 256

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=512, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 8 * 8 * 512

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=1024, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=1024, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 4 * 4 * 1024

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=2048, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=2048, filter_size=(3, 3), pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # 2 * 2 * 2048

    network = lasagne.layers.GlobalPoolLayer(network)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=62,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    inputs = np.array(inputs)

    if targets is None:
        targets = np.zeros((inputs.shape[0]))
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]


def main():
    six.print_('loading data')
    (train_x, train_y, val_x, val_y) = load_data()
    six.print_('load data complete')

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    learning_rate = theano.shared(np.float32(0.01))
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    predict_fn = theano.function([input_var], test_prediction.argmax(axis=1))
    six.print_('build model complete')

    six.print_('start training')

    num_epochs = 10
    batch_size = 64
    for epoch in range(num_epochs):
        if epoch == 50:
            learning_rate.set_value(0.003)
        if epoch == 200:
            learning_rate.set_value(0.0003)
        if epoch == 300:
            learning_rate.set_value(0.00003)

        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(train_x, train_y, batch_size,
                                         shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val_x, val_y, batch_size,
                                         shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(val_x, val_y, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    """predict = np.array([]).astype('int32')
    for batch in iterate_minibatches(test_x, None, batch_size, shuffle=False):
        inputs, targets = batch
        predict_batch = predict_fn(inputs)
        predict = np.append(predict, predict_batch, axis=0)
    save_result(predict)"""


if __name__ == '__main__':
    main()
