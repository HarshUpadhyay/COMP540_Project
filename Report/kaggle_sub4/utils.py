import numpy as np
#import matplotlib.pyplot as plt
import cPickle
import sys
#from skimage.data import imread
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def give_keras_model(img_channels, img_rows, img_cols, nb_classes):

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    '''
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    '''
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    return model


'''
def save_training_data_as_vector(output_file_name, label_data, input_dir, dim_ordering="th", num_samples=50000, do_normalize=False):

    ##
    #   1. reads All png files from disk as a vector of 32x32x3 integer matrices.
    #   2. reads the labels from the given csv file into another vector
    #   3. saves this representation into a new file with name given as input
    ##

    labels = open(label_data)
    labels.readline()

    label_dict = {}

    c = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(len(c)):
        label_dict[c[i]] = i

    print "\nreading data now...\n"
    
    inputFileList = []
    num_samples += 1
    for i in range(1, num_samples):
        fileName = "{}.png".format(i)
        inputFileList.append(fileName)

    X = []
    y = []

    if dim_ordering == "tf":
        for img in inputFileList:
            X.append(imread("{}/{}".format(input_dir, img)))
            y.append(label_dict[labels.readline().strip().split(",")[1]])
            sys.stdout.write("\rReading {}".format(img))
    else:
        for img in inputFileList:
            x = imread("{}/{}".format(input_dir, img))
            X.append(np.array(np.dsplit(x, 3)).reshape(3, 32, 32))
            y.append(label_dict[labels.readline().strip().split(",")[1]])
            sys.stdout.write("\rReading {}".format(img))


    dmp = open(output_file_name, 'w')
    dataValue, dataMean, dataStdDev = preprocess_train_data(np.array(X), do_normalize)
    cPickle.dump({'data': dataValue, 'labels': np.array(y)}, dmp)
    dmp.close()
    labels.close()
    print "\ndone saving dataset vectors.\n"
    return dataMean, dataStdDev


def save_test_data_as_vector(input_dir, ppMean, ppStdDev):

    ##
    #   1. reads All png files from disk as a vector of 32x32x3 integer matrices.
    #   
    #   3. saves this representation into a new file with name given as input
    ##

    fileNumber = 0
    X = []
    imageCount = 1

    inputFileList = []
    for i in range(1,300001):
        fileName = "{}.png".format(i)
        inputFileList.append(fileName)

    for img in inputFileList:

        if imageCount == 50000:
            print "Picking data for file number "+str(fileNumber) + "\n"
            dmp = open("testData/test"+str(fileNumber)+".dat", 'w')
            cPickle.dump({'data': preprocess_test_data(np.array(X), ppMean, ppStdDev)}, dmp)
            dmp.close()
            X = []
            imageCount = 1
            fileNumber = fileNumber + 1
        else:
            #print "HERE"
            #print img
            print "Reading " + str(img) 
            X.append(imread("{}/{}".format(input_dir,img)))
            imageCount = imageCount + 1

    print "Picking data for file number "+str(fileNumber) + "\n"
    dmp = open("testData/test"+str(fileNumber)+".dat", 'w')
    cPickle.dump({'data': preprocess_test_data(np.array(X), ppMean)}, dmp)
    dmp.close()
    
'''

def read_training_data(dat_file_name):
    ##
    #   Reads the pickled image data file and returns the input images and their labels
    ##

    data = open(dat_file_name, 'r')
    data = cPickle.load(data)
    X, y = data['data'], data['labels']
    return X, y


def read_test_data(dat_file_name):
    ##
    #   Reads the pickled image data file and returns the input images and their labels
    ##

    data = open(dat_file_name, 'r')
    data = cPickle.load(data)
    X = data['data']
    return X




# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
"""
def visualize_data(X_train, y_train, num_samples):
    classes = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = num_samples
    for y, cls in enumerate(classes):
        indices = np.flatnonzero(y_train == y)
        indices = np.random.choice(indices, samples_per_class, replace=False)
        for i, idx in enumerate(indices):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.savefig('CNN_samples.pdf')
    plt.close()
"""

# subsampling  the data
def subsample(num_training, num_validation, num_test, X_train, y_train, X_test, y_test):
    # Our validation set will be num_validation points from the original
    # training set.

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # We use the first num_test points of the original test set as our
    # test set.

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_train_data(train_data, do_normalize = True):
    ##
    #   :param test_data: input vector of mx32x32x3 images
    #   :return:
    ##

    # flattening the data into a row
    # test_data = np.reshape(test_data, (test_data.shape[0], -1))
    meanValue = np.mean(train_data, axis=0)
    stdDev = np.std(train_data,  axis=0)
    if do_normalize:
        sys.stdout.write("\nnormalizing. No Whitening done.\r")
        # mean subtraction
        train_data = train_data - meanValue
        # standard deviation normalization
        train_data = train_data / stdDev
   
    """
    #   SVD whitening
    cov = np.dot(test_data.T, test_data)/ test_data.shape[0]
    U,S,V = np.linalg.svd(cov)
    test_data_rot = np.dot(test_data, U)
    """
    train_data = train_data.astype(np.float32)
    sys.stdout.write("\r\t\t...done!\n")
    return train_data, meanValue, stdDev

def preprocess_test_data(test_data, ppMean, ppStdDev):
    ##
    #   :param test_data: input vector of mx32x32x3 images
    #   :return:
    ##
    print "\nnormalizing data now...\n"
    # flattening the data into a row
    # test_data = np.reshape(test_data, (test_data.shape[0], -1))

    # mean subtraction
    test_data = test_data - ppMean

    # standard deviation normalization
    test_data = test_data / ppStdDev

    test_data = test_data.astype(np.float32)
    '''
    #   SVD whitening
    cov = np.dot(test_data.T, test_data)/ test_data.shape[0]
    U,S,V = np.linalg.svd(cov)
    test_data_rot = np.dot(test_data, U)
    '''
    return test_data

