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
import scipy.fftpack as fft
import theano.tensor as T

settings = cfg.getConfig()
audioData = openWavFile(settings['source'])
phaseData, meanPhiWav, stdPhiWav = extractWaveData(audioData, settings['section-count'], blkCount=settings['block-count'], returnObj="phase", olapf=settings['overlap'])
magntData, maxMagWav, minMagWav = extractWaveData(audioData, settings['section-count'], blkCount=settings['block-count'], returnObj="magnitude", olapf=settings['overlap'])

px = phaseData.shape[0]
py = phaseData.shape[1]
pz = phaseData.shape[2]

layerCount = settings['layer-count']

kindex = 0
phiEncoderModel = Sequential()    
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    decoderWeight = np.load('./autoencoder-weights/' + settings['phase-encoder'] + '-phase-decoder-' + str(encInd) + '.npy')
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear')
    phiEncoderModel.add(timeDistDense)
    phiEncoderModel.layers[encIndM].set_weights(decoderWeight)

magEncoderModel = Sequential()
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    decoderWeight = np.load('./autoencoder-weights/' + settings['magnitude-encoder'] + '-magnitude-decoder-' + str(encInd) + '.npy')
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear')
    magEncoderModel.add(timeDistDense)
    magEncoderModel.layers[encIndM].set_weights(decoderWeight)

phiEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')
magEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')

nphaseData = np.load('./coded-data/' + settings['coded-file'] + '-phase-result.npy')
nmagntData = np.load('./coded-data/' + settings['coded-file'] + '-magnitude-result.npy')
nphaseData = np.reshape(nphaseData, (settings['section-count'], -1, nphaseData.shape[1]))
nmagntData = np.reshape(nmagntData, (settings['section-count'], -1, nmagntData.shape[1]))
kPhaseData = phiEncoderModel.predict(nphaseData)
kMagnData = magEncoderModel.predict(nmagntData)

np.save('./results-data/' + settings['phase-result'] + 'phase-data', kPhaseData)
np.save('./results-data/' + settings['magnitude-result'] + 'magnitude-data', kMagnData)
