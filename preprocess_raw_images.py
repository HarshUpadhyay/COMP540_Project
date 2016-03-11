import numpy as np
import cPickle as pickle

from skimage.data import imread

labels = open("trainLabels.csv")
labels.readline()

label_dict={}
c = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(len(c)):
    label_dict[c[i]] = i

y = []
X= []

for i in xrange(50000):
    X.append(imread("train/{}.png".format(i+1)))
    y.append(label_dict[labels.readline().strip().split(",")[1]])

X = np.array(X)
y = np.array(y)

dump = open('pickle_dump.dat', 'w')
data = pickle.dump({'data': X, 'labels': y}, dump)
dump.close()
labels.close()
#data = open('pickle_dump.dat', 'r')
#data = pickle.load(data)
#X, y = data['data'], data['labels']