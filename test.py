import utils
import image_utils
import matplotlib.pyplot as plt
import cPickle


utils.save_training_data_as_vector("tmp.dat", "trainLabels.csv", "tmp")
X,y=utils.read_training_data('tmp.dat')
plt.imshow(X[0].astype('uint8'))
plt.savefig('tmp_img.pdf')
plt.close()