import utils
from sknn.mlp import Classifier, Convolution, Layer
import numpy as np
from os import path
import image_utils

dataset = "train1.dat"
if path.isfile(dataset) == False:
    utils.save_training_data_as_vector(dataset, "trainLabels.csv", "train")


print "Begin training by reading the pickled dataset"
X, y = utils.read_training_data(dataset)

print "shape of the dataset and  labels: x={} y={}\n".format(X.shape, y.shape)
Xtrain, ytrain = X[:49000], y[:49000]
Xval, yval = X[49000:], y[49000:]

print "Training size = {}".format(len(ytrain))
print "Validation size = {}".format(len(yval))

nn = Classifier(
    layers=[
        Convolution("Rectifier", name="layer1", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full", pool_shape(2,2), pool_type="max", dropout=0.25),
        Convolution("Rectifier", name="layer1", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full", pool_shape(2,2), pool_type="max", dropout=0.25),
        Convolution("Rectifier", name="layer1", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full", pool_shape(2,2), pool_type="max", dropout=0.25),
    ],
    learning_rate=0.02,
    n_iter=100,
    regularize="L2")
]
nn.fit(Xtrain, ytrain)

pred = nn.predict(Xval)

accuracy = np.mean((yval == pred).astype(float)) * 100

print "accuracy on validation data: {}%".format(accuracy)
