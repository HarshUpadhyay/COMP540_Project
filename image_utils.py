import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import os


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    :param num_training
    :param num_validation
    :param num_test

    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the softmax classifier.
    """
    # Load the raw CIFAR-10 data

    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # subsample the data

    X_train, y_train, X_val, y_val, X_test, y_test = subsample(num_training, num_validation, num_test,
                                                               X_train, y_train,
                                                               X_test, y_test)

    # visualize a subset of the training data

    visualize_cifar10(X_train, y_train)

    # preprocess data

    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_CIFAR_batch(filename):
    """
     :param filename
    load single batch of cifar
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(cifar10_root):
    """
    :param cifar10_root
    load all of cifar
    """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(cifar10_root, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(cifar10_root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def preprocess(X_train, X_val, X_test):
    # Preprocessing: reshape the image data into rows

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # As a sanity check, print out the shapes of the data
    print 'Training data shape: ', X_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Test data shape: ', X_test.shape

    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data

    mean_image = np.mean(X_train, axis=0)
    #  plt.figure(figsize=(4,4))
    #  plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image

    # second: subtract the mean image from train and test data

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # third: append the bias dimension of ones (i.e. bias trick) so that our softmax regressor
    # only has to worry about optimizing a single weight matrix theta.
    # Also, lets transform data matrices so that each image is a row.

    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    print 'Training data shape with bias term: ', X_train.shape
    print 'Validation data shape with bias term: ', X_val.shape
    print 'Test data shape with bias term: ', X_test.shape

    return X_train, X_val, X_test
