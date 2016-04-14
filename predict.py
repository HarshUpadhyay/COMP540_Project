import utils
import numpy as np
from keras.models import model_from_json

#dat_file_name = "test.dat"

#X_test = utils.read_test_data(dat_file_name)
X, y = utils.read_training_data("train_keras_tmp.dat")
split = 0.98
mask = int(X.shape[0] * split)

#this part is important. If input is not preprocessed same way as the test the accuracy will decrease
X /= 255

#selecting subset
X_test, y_test = X[mask:], y[mask:]
model = model_from_json(open('keras_arch0.json').read())
model.load_weights('keras_weights0.h5')

pred = model.predict(X_test, verbose=1)
pred = np.argmax(pred, axis=1)

accuracy = np.mean((y_test == pred).astype(float)) * 100
str = "##########################################"
print "\n{}\naccuracy on validation data: {}%\n{}\n\n".format(str,accuracy,str)
