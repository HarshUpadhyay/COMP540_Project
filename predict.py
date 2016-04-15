import utils
import numpy as np
from keras.optimizers import SGD

print "Running prediction Script now..."
print "Reading test dataset"

#dat_file_name = "test.dat"

#X_test = utils.read_test_data(dat_file_name)
X, y = utils.read_training_data("train_keras_tmp.dat")
split = 0.98
mask = int(X.shape[0] * split)

#this part is important. If input is not preprocessed same way as the test the accuracy will decrease
X /= 255
nb_classes = 10
# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3
#selecting subset
X_test, y_test = X[mask:], y[mask:]
print "done"
print "Loading saved Model from disk"
model = utils.give_keras_model(img_channels, img_rows, img_cols, nb_classes)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
model.load_weights('keras_weights_tmp.h5')

print "Done! Now predicting on Test Data"
pred = model.predict(X_test, verbose=1)
pred = np.argmax(pred, axis=1)

accuracy = np.mean((y_test == pred).astype(float)) * 100
str = "##########################################"
print "\n{}\naccuracy on validation data: {}%\n{}\n\n".format(str,accuracy,str)
