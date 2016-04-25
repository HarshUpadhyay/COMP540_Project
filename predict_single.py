import utils
import numpy as np
from keras.optimizers import SGD
from skimage.data import imread

print "Running prediction Script now..."
print "Reading test dataset"

#dat_file_name = "test.dat"
#X_test = utils.read_test_data(dat_file_name)
X_test=[]
for i in range(1, 300001):
    x = imread("test/{}.png".format(i))
    X_test.append(np.array(np.dsplit(x, 3)).reshape(3, 32, 32))

#this part is important. If input is not preprocessed same way as the test the accuracy will decrease
X_test = np.asarray(X_test)
X_test = X_test.astype(np.float32)
X_test /= 255
c = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
nb_classes = len(c)

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3
#selecting subset
print "done"
print "Loading saved Model from disk"
model = utils.give_keras_model(img_channels, img_rows, img_cols, nb_classes)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
model.load_weights('wt_ensemble_member2.h5')

print "Done! Now predicting on Test Data"
pred = model.predict_classes(X_test)

'''
accuracy = np.mean((y_test == pred).astype(float)) * 100
str = "##########################################"
print "\n{}\naccuracy on validation data: {}%\n{}\n\n".format(str,accuracy,str)
'''

result = open("predictions.csv", mode='w')
result.write("id,label\n")

for i, p in enumerate(pred):
    result.write("{},{}\n".format(i+1, c[p]))

result.close()
