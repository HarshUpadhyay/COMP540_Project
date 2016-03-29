from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import image_utils
from skimage.data import imread

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

#X_train, y_train, X_val, y_val, X_test, y_test = image_utils.get_CIFAR10_data()
y = imread("")
print("done")

#Just checking github. 
