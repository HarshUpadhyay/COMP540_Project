from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import utils
from os import path

dataset = "train_keras.dat"
if path.isfile(dataset) == False:
    utils.save_training_data_as_vector(dataset, "trainLabels.csv", "train")


print "Begin training by reading the pickled dataset"
X, y = utils.read_training_data(dataset)

print "shape of the dataset and  labels: x={} y={}\n".format(X.shape, y.shape)
X_train, y_train = X[:49000], y[:49000]
X_val, y_val = X[49000:], y[49000:]

print "Training size = {}".format(len(y_train))
print "Validation size = {}".format(len(y_val))


model = Sequential()
# input: 32x32 images with 3 channels -> (32, 32, 3) tensors.
# this applies 10 convolution filters of size 3x3 each.
model.add(Convolution2D(10, 3, 3, border_mode='valid',input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(Convolution2D(10, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=25000, nb_epoch=10, verbose=1, validation_split=0.06, show_accuracy=True,)
