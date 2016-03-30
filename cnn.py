import utils
from sknn.mlp import Classifier, Convolution, Layer
import numpy as np
import image_utils


#utils.save_training_data_as_vector("test.dat", "trainLabels.csv", "test")

X, y = utils.read_training_data('train.dat')

#print len(X), len(y)

Xtrain, Xval = X[:40000], X[40000:]
ytrain, yval = X[:40000], y[40000:]

#print Xtrain[0][0][0]

nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
        Layer("Softmax")],
    n_iter=1000)
nn.fit(Xtrain, ytrain)

pred = nn.predict(Xval)

accuracy = np.mean((yval == pred).astype(float)) * 100

print "accuracy: {}%".format(accuracy)
