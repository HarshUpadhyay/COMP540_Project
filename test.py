import utils
import image_utils
import matplotlib.pyplot as plt
import cPickle


utils.save_training_data_as_vector("train.dat", "trainLabels.csv", "train")
print "Hello"
X,y=utils.read_training_data('train.dat')

print len(X), len(y)
#plt.imshow(X[0].astype('uint8'))
#plt.savefig('tmp_img.pdf')
#plt.close()
