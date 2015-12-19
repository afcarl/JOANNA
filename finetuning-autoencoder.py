from keras.layers.core import TimeDistributedDense, Merge, AutoEncoderDropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import ELU
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, AutoEncoder, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
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
print phaseData.shape
print magntData.shape
px = phaseData.shape[0]
py = phaseData.shape[1]
pz = phaseData.shape[2]
print phaseData.shape
print magntData.shape
layerCount = settings['layer-count']

last = 0
phiEncoderModel = Sequential()  
for encInd in range(0, layerCount):
    midlay = (encInd) * settings['dim-decrease']
    outlay = (encInd + 1) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, input_length=py)
    if encInd > 0:
        phiEncoderModel.add(AutoEncoderDropout(0.2))
    phiEncoderModel.add(timeDistDense)
    phiEncoderModel.add(ELU())
    last = pz - outlay

phiEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
phiEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))

magEncoderModel = Sequential()    
for encInd in range(0, layerCount):
    midlay = (encInd) * settings['dim-decrease']
    outlay = (encInd + 1) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, input_length=py)
    if encInd > 0:
        magEncoderModel.add(AutoEncoderDropout(0.2))
    magEncoderModel.add(timeDistDense)
    magEncoderModel.add(ELU())
    last = pz - outlay

magEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
magEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))

print len(magEncoderModel.layers)
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear')
    phiEncoderModel.add(timeDistDense)

if os.path.isfile('./autoencoder-weights/' + settings['phase-encoder'] + '-phase-AE') and settings['load-weights']:
    phiEncoderModel.load_weights('./autoencoder-weights/' + settings['phase-encoder'] + '-phase-AE')
phiEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')
  
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear')
    magEncoderModel.add(timeDistDense)

if os.path.isfile('./autoencoder-weights/' + settings['magnitude-encoder'] + '-magnitude-AE') and settings['load-weights']:
    magEncoderModel.load_weights('./autoencoder-weights/' + settings['magnitude-encoder'] + '-magnitude-AE')
magEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')

print 'start'
for knum in xrange(settings['ae-iteration']):
    phiEncoderModel.fit(phaseData, phaseData, batch_size=1, nb_epoch=settings['ae-epoch'], verbose=0, validation_split=0.0, shuffle=False)
    startLoss = phiEncoderModel.train_on_batch(phaseData[:1], phaseData[:1])
    print 'phase loss: ' + str(startLoss)
    magEncoderModel.fit(magntData, magntData, batch_size=1, nb_epoch=settings['ae-epoch'], verbose=0, validation_split=0.0, shuffle=False)
    mstartLoss = magEncoderModel.train_on_batch(magntData[:1], magntData[:1])
    print 'magnitude loss: ' + str(mstartLoss)
    phiEncoderModel.save_weights('./autoencoder-weights/' + settings['phase-encoder'] + '-phase-AE', overwrite=True)
    magEncoderModel.save_weights('./autoencoder-weights/' + settings['magnitude-encoder'] + '-magnitude-AE', overwrite=True)
