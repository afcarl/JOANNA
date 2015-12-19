from keras.layers.core import TimeDistributedDense, Merge, AutoEncoderDropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.advanced_activations import ELU
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
import scipy.fftpack as fft

def modelWeightsLoader(model, filepath, loadLayers=[]):
    # Loads weights from HDF5 file
    import h5py
    f = h5py.File(filepath)
    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if k in loadLayers or not loadLayers:
            model.layers[k].set_weights(weights)
    f.close()
    return model
     

settings = cfg.getConfig()
audioData = openWavFile(settings['source'])
phaseData, meanPhiWav, stdPhiWav = extractWaveData(audioData, settings['section-count'], blkCount=settings['block-count'], returnObj="phase", olapf=settings['overlap'])
magntData, maxMagWav, minMagWav = extractWaveData(audioData, settings['section-count'], blkCount=settings['block-count'], returnObj="magnitude", olapf=settings['overlap'])

px = phaseData.shape[0]
py = phaseData.shape[1]
pz = phaseData.shape[2]

lastp = 0
lastm = 0

layerCount = settings['layer-count']
phiEncoderModel = Sequential()    
for encInd in range(0, layerCount):
    midlay = (encInd) * settings['dim-decrease']
    outlay = (encInd + 1) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, input_length=py)
    if encInd > 0:
        phiEncoderModel.add(AutoEncoderDropout(0.0))
    layCount = len(phiEncoderModel.layers)
    phiEncoderModel.add(timeDistDense)
    phiEncoderModel.add(ELU())
    lastp = pz - outlay

magEncoderModel = Sequential()    
for encInd in range(0, layerCount):
    midlay = (encInd) * settings['dim-decrease']
    outlay = (encInd + 1) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, input_length=py)
    if encInd > 0:
        magEncoderModel.add(AutoEncoderDropout(0.0))
    layCount = len(magEncoderModel.layers)
    magEncoderModel.add(timeDistDense)
    magEncoderModel.add(ELU())
    lastm = pz - outlay	

magEncoderModel.add(LSTM(input_dim=lastm, output_dim=lastm, return_sequences=True))
phiEncoderModel.add(LSTM(input_dim=lastp, output_dim=lastp, return_sequences=True))
print len(magEncoderModel.layers)
print len(phiEncoderModel.layers)
magEncoderModel = modelWeightsLoader(magEncoderModel, './autoencoder-weights/' + settings['magnitude-encoder'] + '-magnitude-AE', range(len(magEncoderModel.layers)))
phiEncoderModel = modelWeightsLoader(phiEncoderModel, './autoencoder-weights/' + settings['phase-encoder'] + '-phase-AE', range(len(phiEncoderModel.layers)))

magEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')
phiEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')

kPhaseDataI = phiEncoderModel.predict(phaseData)
kMagnDataI = magEncoderModel.predict(magntData)

np.save('./coded-data/' + settings['coded-file'] + '-phase-o', kPhaseDataI)
np.save('./coded-data/' + settings['coded-file'] + '-magnitude-o', kMagnDataI)
