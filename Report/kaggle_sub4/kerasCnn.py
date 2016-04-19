import utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from os import path

dataset = "train_keras_noNorm.dat"
model_weights_name = "keras_weights"
split = 0.98

if path.isfile(dataset) == False:
    utils.save_training_data_as_vector(dataset, "trainLabels.csv", "train", )


print "Begin training by reading the pickled dataset"
X, y = utils.read_training_data(dataset)
mean_img = np.mean(X, axis = 0)
print "mean image: ", mean_img.shape

X -= mean_img
X /= 255

print "shape of the dataset and  labels: x={} y={}\n".format(X.shape, y.shape)
mask = int(X.shape[0] * split)
X_train, y_train = X[:mask], y[:mask]
X_val, y_val = X[mask:], y[mask:]

print "Training size = {}".format(len(y_train))
print "Validation size = {}".format(len(y_val))

batch_size = 32
nb_classes = 10
nb_epoch = 300
nb_dataModelRun = 1

data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)


model = utils.give_keras_model(img_channels, img_rows, img_cols, nb_classes)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_val, Y_val), shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)


    #fit the model on the batches genreated by datagen.flow()
    # Run the iterations for nb_dataModelRun times
    # Save the model per nb_epoch number of iterations of model fitting 
    # Alternately saving model in two files
    # Use previously used model as seed to current model

    # index for nb_dataModelRun
    i = 0
    # index for Model Weight file
    flag = 0
    for i in range(nb_dataModelRun):
        if i!=0:
            
            # load model weights from previous round
            model.load_weights("{}{}.h5".format(model_weights_name, flag))
            # reset flag
            flag = (flag + 1)%2

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val),
                            nb_worker=1)
        #save model
        #print "saving model..."
        model.save_weights("{}{}.h5".format(model_weights_name, flag),overwrite=True)
        #print "\nDone saving model.\n " \
            #"Here they are:\n" \
            #"Model Weights {}\n"\
            #.format(model_weights_name)


    print "\nnow, lets do some prediction on the validation set!\n"
    pred = model.predict(X_val, verbose=1).astype(float)
    pred = np.argmax(pred, axis=1)

    accuracy = np.mean((y_val == pred).astype(float)) * 100
    str = "##########################################"
    print "\n{}\naccuracy on validation data: {}%\n{}\n\n".format(str,accuracy,str)

   
