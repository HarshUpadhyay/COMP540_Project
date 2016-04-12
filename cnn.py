import utils
from sknn.mlp import Classifier, Convolution, Layer
import numpy as np
from os import path
import image_utils

dataset = "train_cnn.dat"
if path.isfile(dataset) == False:
    utils.save_training_data_as_vector(dataset, "trainLabels.csv", "train", dim_ordering="tf", do_normalize=True)


print "Begin training by reading the pickled dataset"
X, y = utils.read_training_data(dataset)

print "shape of the dataset and  labels: x={} y={}\n".format(X.shape, y.shape)
Xtrain, ytrain = X[:45000], y[:45000]
Xval, yval = X[45000:], y[45000:]

print "Training size = {}".format(len(ytrain))
print "Validation size = {}".format(len(yval))

nn = Classifier(
    layers=[
        Convolution("Rectifier", name="layer1", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full"),
        Convolution("Rectifier", name="layer2", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full", pool_shape=(2,2), pool_type="max"),
        Convolution("Rectifier", name="layer3", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full"),
        Convolution("Rectifier", name="layer4", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full", pool_shape=(2,2), pool_type="max"),
        Convolution("Rectifier", name="layer5", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full"),
        Convolution("Rectifier", name="layer6", channels=8, kernel_shape=(3,3), kernel_stride=(1,1), border_mode="full", pool_shape=(2,2), pool_type="max"),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=500,
    regularize="L2",
    verbose=True)

nn.fit(Xtrain, ytrain)

pred = nn.predict(Xval)

accuracy = np.mean((yval == pred).astype(float)) * 100
str = "##########################################"
print "\n{}\naccuracy on validation data: {}%\n{}\n\n".format(str,accuracy,str)
