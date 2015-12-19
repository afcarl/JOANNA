from keras.layers.core import TimeDistributedDense, Merge
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, AutoEncoder
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import containers
from utils import *
import config as cfg
import wave, struct, time
import numpy as np
import os
import scipy.io.wavfile as wav   

def modelWeightsLoader(model, filepath, loadLayers={}):
    # Loads weights from HDF5 file
    import h5py
    f = h5py.File(filepath)
    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if k in loadLayers:
            model.layers[loadLayers[k]].set_weights(weights)
        elif not loadLayers:
            model.layers[k].set_weights(weights)
    f.close()
    return model

settings = cfg.getConfig()

kMagnData = np.load('./coded-data/' + settings['coded-file'] + '-magnitude-i.npy')
kPhaseData = np.load('./coded-data/' + settings['coded-file'] + '-phase-i.npy')

kMagnDataO = np.load('./coded-data/' + settings['coded-file'] + '-magnitude-o.npy')
kPhaseDataO = np.load('./coded-data/' + settings['coded-file'] + '-phase-o.npy')
yMagnConcat = blockShift(kMagnDataO, shift=1)
yConcat = blockShift(kPhaseDataO, shift=1)

yx = kMagnData.shape[0]
yy = kMagnData.shape[1]
yz = kMagnData.shape[2]

ox = yConcat.shape[0]
oy = yConcat.shape[1]
oz = yConcat.shape[2]

phiModel = Sequential()
print yConcat.shape

phiModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
phiModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
phiModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
phiModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
phiLstmLayer1 = LSTM(input_dim=yz, output_dim=yz, return_sequences=True)
phiLstmLayer1.trainable = False
phiModel.add(phiLstmLayer1)

if os.path.isfile('./lstm-weights/phi-' + settings['lstm-file']):
    phiModel.load_weights('./lstm-weights/phi-' + settings['lstm-file'])

phiModel = modelWeightsLoader(phiModel, './autoencoder-weights/' + settings['phase-encoder'] + '-phase-AE', {18:4})
phiModel.compile(loss='mean_squared_error', optimizer='rmsprop')

magnModel = Sequential()
magnModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
magnModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
magnModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
magnModel.add(LSTM(input_dim=yz, output_dim=yz, return_sequences=True))
magLstmLayer1 = LSTM(input_dim=yz, output_dim=yz, return_sequences=True)
magLstmLayer1.trainable = False
magnModel.add(magLstmLayer1)

if os.path.isfile('./lstm-weights/mag-' + settings['lstm-file']):
    magnModel.load_weights('./lstm-weights/mag-' + settings['lstm-file'])

magnModel = modelWeightsLoader(magnModel, './autoencoder-weights/' + settings['magnitude-encoder'] + '-magnitude-AE', {18:4})
magnModel.compile(loss='mean_squared_error', optimizer='rmsprop')

pLowLoss = 999
mLowLoss = 999
for j in xrange(settings['lstm-iteration']):
    start = time.time()
    lnum = 1
    phiModel.fit(kPhaseData, yConcat, batch_size=lnum, nb_epoch=settings['lstm-epoch'], verbose=0, validation_split=0.0)
    phiModel.save_weights('./lstm-weights/phi-' + settings['lstm-file'], overwrite=True)
    magnModel.fit(kMagnData, yMagnConcat, batch_size=lnum, nb_epoch=settings['lstm-epoch'], verbose=0, validation_split=0.0)
    magnModel.save_weights('./lstm-weights/mag-' + settings['lstm-file'], overwrite=True)
    pStartLoss = phiModel.train_on_batch(kPhaseData[:1], yConcat[:1])
    mStartLoss = magnModel.train_on_batch(kMagnData[:1], yMagnConcat[:1])
    pLowLoss = pStartLoss if pStartLoss < pLowLoss else pLowLoss
    mLowLoss = mStartLoss if mStartLoss < mLowLoss else mLowLoss
    end = time.time()
    timeDiff = end - start
    print str(timeDiff) + ' - phase loss: ' + str(pStartLoss) + ' (' + str(pLowLoss) + '), magnitude loss: ' + str(mStartLoss) + ' (' + str(mLowLoss) + ')'

