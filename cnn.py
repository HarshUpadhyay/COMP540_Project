import utils
from sknn.mlp import Classifier, Convolution, Layer
import numpy as np
import image_utils


#utils.save_training_data_as_vector("tmp.dat", "trainLabels.csv", "tmp")

X, y = utils.read_training_data('train.dat')

print "size of the tmp dataset: x={} y={}\n".format(len(X), len(y))

Xtrain, Xval = X[:40000], X[40000:]
ytrain, yval = y[:40000], y[40000:]
#Xtrain, Xval = X[:900], X[900:]
#ytrain, yval = y[:900], y[900:]


nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=8, kernel_shape=(3,3)),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    regularize="L2")
nn.fit(Xtrain, ytrain)

pred = nn.predict(Xval)

accuracy = np.mean((yval == pred).astype(float)) * 100

print "accuracy: {}%".format(accuracy)
