import utils
from sknn.mlp import Classifier, Convolution, Layer
import numpy as np
import image_utils


utils.save_training_data_as_vector("train.dat", "trainLabels.csv", "train")

print "reading pickled data"

X, y = utils.read_training_data('train.dat')

Xtrain, Xval = X[0:49000], X[49000:50000]
ytrain, yval = y[0:49000], y[49000:50000]

nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
        Layer("Softmax")],
    n_iter=5)

print "training"

nn.fit(Xtrain, ytrain)

pred = nn.predict(Xval)

accuracy = np.mean((yval == pred).astype(float)) * 100

print "accuracy on validation set: {}%".format(accuracy)
