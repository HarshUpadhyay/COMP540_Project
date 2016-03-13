import numpy as np
import cPickle as pickle

from skimage.data import imread


def save_training_data_as_vector(output_file_name, label_data, input_dir ):

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

    y = []
    X = []

    for i in xrange(len(y)):
        X.append(imread("{}/{}.png".format(input_dir, i+1)))
        y.append(label_dict[labels.readline().strip().split(",")[1]])

    X = np.array(X)
    y = np.array(y)

    dmp = open(output_file_name, 'w')
    pickle.dump({'data': X, 'labels': y}, dmp)
    dmp.close()
    labels.close()


def read_img_data(dat_file_name):
    ##
    #   Reads the pickled image data file and returns the input images and their labels
    ##

    data = open(dat_file_name, 'r')
    data = pickle.load(data)
    X, y = data['data'], data['labels']
    return X, y

