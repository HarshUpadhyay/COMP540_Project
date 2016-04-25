import utils
import pylab as pl
import numpy as np
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm


'''
dat_file_name = "train_keras_tmp.dat"
X, y = utils.read_training_data(dat_file_name)
X = X.astype(np.float32)
X -= np.mean(X)
X /=255
'''
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border,
                            3),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


##################################################################################
img_rows, img_cols = 32, 32
img_channels = 3
nb_classes = 10

M = utils.give_keras_model(img_channels, img_rows, img_cols, nb_classes)
M.load_weights('wt_ensemble_member2.h5')

'''
layer_op = []

for layer in model.layers[1:]:
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output])
    layer_output = get_layer_output([X,0])[0]
    layer_op.append(layer_output)

layer_op = np.asarray(layer_op)
'''
W = M.layers[8].W.get_value(borrow=True)
W = np.squeeze(W)
print("W shape : ", W.shape)

pl.figure(figsize=(15, 15))
pl.title('Conv Layer 4 weights')
nice_imshow(pl.gca(), make_mosaic(W, 8, 8), cmap=cm.binary)
pl.savefig('ConvL4.pdf')
pl.close()