import utils
from sknn.mlp import Classifier, Convolution, Layer
import numpy as np
import image_utils


utils.save_training_data_as_vector("test.dat", "trainLabels.csv", "test")
'''
X, y = utils.read_training_data('')

nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
        Layer("Softmax")],
        regularize="L2",
    n_iter=1000)
nn.fit(X, y)

pred = nn.predict(X)

accuracy = np.mean((y == pred).astype(float)) * 100

print "accuracy: {}".format(accuracy)
'''