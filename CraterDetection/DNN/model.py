#Written by Yarin Gal
#yg279@cam.ac.uk‎

from neon.data import ArrayIterator
from neon import backends
from neon.transforms import Rectlin
from neon.initializers import Constant, Xavier, GlorotUniform
from neon.backends import gen_backend
from neon.models import Model
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
from neon.optimizers import Adam
from neon.callbacks.callbacks import Callbacks
from neon.layers import Affine, Dropout, Conv, Pooling
from neon.transforms import Softmax
from neon.transforms import Misclassification

#Depending on system, can also use gpu
be = gen_backend(backend='cpu', batch_size=64)


def get_neon_set(X, y):
    s = ArrayIterator(X, y=y, lshape=(1, 32, 32), nclass=2)
    # for x, t in s:
    #     print x.get(), t.get() #Object, Label
    #     break
    return s


def fit_model(train_set, val_set, num_epochs=50):
    relu = Rectlin()
    conv_params = {'strides': 1,
                   'padding': 1,
                   'init': Xavier(local=True),  # Xavier sqrt(3)/num_inputs [CHECK THIS]
                   'bias': Constant(0),
                   'activation': relu}

    layers = []
    layers.append(Conv((3, 3, 128), **conv_params))  # 3x3 kernel * 128 nodes
    layers.append(Pooling(2))
    layers.append(Conv((3, 3, 128), **conv_params))
    layers.append(Pooling(2))  # Highest value from 2x2 window.
    layers.append(Conv((3, 3, 128), **conv_params))
    layers.append(
        Dropout(keep=0.5))  # Flattens Convolution into a flat array, with probability 0.5 sets activation values to 0
    layers.append(Affine(nout=128, init=GlorotUniform(), bias=Constant(0),
                         activation=relu))  # 1 value per conv kernel - Linear Combination of layers
    layers.append(Dropout(keep=0.5))
    layers.append(Affine(nout=2, init=GlorotUniform(), bias=Constant(0), activation=Softmax(), name="class_layer"))

    # initialize model object
    cnn = Model(layers=layers)
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())

    optimizer = Adam()

    # callbacks = Callbacks(cnn)
    #    out_fname = 'yarin_fdl_out_data.h5'
    callbacks = Callbacks(cnn, eval_set=val_set, eval_freq=1)  # , output_file=out_fname

    cnn.fit(train_set, optimizer=optimizer, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

    return cnn


def test_model(model, s):
    error = model.eval(s, metric=Misclassification()) * 100
    return error
