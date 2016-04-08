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
        Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
        Layer("Softmax")],
    learning_rate=0.02,
<<<<<<< HEAD
    n_iter=50)
=======
    n_iter=100,
    regularize="L2")
>>>>>>> 60218740f0c36cbe6b58dfcea82ad37c6e24ed5b
nn.fit(Xtrain, ytrain)

pred = nn.predict(Xval)

accuracy = np.mean((yval == pred).astype(float)) * 100

print "accuracy on validation data: {}%".format(accuracy)
