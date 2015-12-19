from keras.layers.core import TimeDistributedDense, Merge, AutoEncoderDropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import PReLU, LeakyReLU, ParametricSoftplus
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
phaseData, meanPhiWav, stdPhiWav = extractWaveData(audioData, settings['section-count'], blkCount=251, returnObj="phase", olapf=settings['overlap'])
magntData, maxMagWav, minMagWav = extractWaveData(audioData, settings['section-count'], blkCount=251, returnObj="magnitude", olapf=settings['overlap'])
print phaseData.shape
print magntData.shape
px = phaseData.shape[0]
py = phaseData.shape[1]
pz = phaseData.shape[2]
settings['dim-decrease'] = 300
layerCount = 2
phiEncIndex = [] 
phiDecIndex = [] 
magEncIndex = [] 
magDecIndex = [] 

phiEncoderModel = Sequential()
phiEncoderModel.add(TimeDistributedDense(input_dim=pz, output_dim=pz, activation='relu'))
for encInd in range(0, layerCount):
    midlay = (encInd) * settings['dim-decrease']
    outlay = (encInd + 1) * settings['dim-decrease']
    timeDistDense = LSTM(input_dim=pz - midlay, output_dim=pz - outlay, return_sequences=True)
    layCount = len(phiEncoderModel.layers)
    phiEncoderModel.add(AutoEncoderDropout(0.2))
    layCount = len(phiEncoderModel.layers)
    phiEncIndex.append(layCount)
    phiEncoderModel.add(timeDistDense)

magEncoderModel = Sequential()    
magEncoderModel.add(TimeDistributedDense(input_dim=pz, output_dim=pz, activation='relu'))
for encInd in range(0, layerCount):
    midlay = (encInd) * settings['dim-decrease']
    outlay = (encInd + 1) * settings['dim-decrease']
    timeDistDense = LSTM(input_dim=pz - midlay, output_dim=pz - outlay, return_sequences=True)
    layCount = len(magEncoderModel.layers)
    magEncoderModel.add(AutoEncoderDropout(0.2))
    layCount = len(magEncoderModel.layers)
    magEncIndex.append(layCount)
    magEncoderModel.add(timeDistDense)
  
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    timeDistDense = LSTM(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear', return_sequences=True)
    layCount = len(phiEncoderModel.layers)
    phiDecIndex.append(layCount)
    phiEncoderModel.add(timeDistDense)
	
phiEncoderModel.add(TimeDistributedDense(input_dim=pz, output_dim=pz, activation='linear'))
if os.path.isfile('./lstmae-weights/lstmae-phase-AE'):
    phiEncoderModel.load_weights('./lstmae-weights/lstmae-phase-AE')
phiEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')
  
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    timeDistDense = LSTM(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear', return_sequences=True)
    layCount = len(magEncoderModel.layers)
    magDecIndex.append(layCount)
    magEncoderModel.add(timeDistDense)
	
magEncoderModel.add(TimeDistributedDense(input_dim=pz, output_dim=pz, activation='linear'))
if os.path.isfile('./lstmae-weights/lstmae-magnitude-AE'):
    magEncoderModel.load_weights('./lstmae-weights/lstmae-magnitude-AE')
magEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')

print phiEncIndex
print phiDecIndex
print 'start'
for knum in xrange(settings['ae-iteration']):
    phiEncoderModel.fit(phaseData, phaseData, batch_size=1, nb_epoch=settings['ae-epoch'], verbose=1, validation_split=0.0, shuffle=False)
    startLoss = phiEncoderModel.train_on_batch(phaseData, phaseData)
    print 'phase loss: ' + str(startLoss)
    magEncoderModel.fit(magntData, magntData, batch_size=1, nb_epoch=settings['ae-epoch'], verbose=1, validation_split=0.0, shuffle=False)
    mstartLoss = magEncoderModel.train_on_batch(magntData, magntData)
    print 'magnitude loss: ' + str(mstartLoss)
    phiEncoderModel.save_weights('./lstmae-weights/lstmae-phase-AE', overwrite=True)
    magEncoderModel.save_weights('./lstmae-weights/lstmae-magnitude-AE', overwrite=True)
mec = 0
mdc = 0
pec = 0
pdc = 0
for pCount in xrange(len(phiEncoderModel.layers)):
    if pCount in phiEncIndex:
        np.save('./lstmae-weights/lstmae-phase-encoder-' + str(pec), phiEncoderModel.layers[pCount].get_weights())
        pec += 1
    elif pCount in phiDecIndex:
        k = (len(phiDecIndex) - 1) - pdc
        np.save('./lstmae-weights/lstmae-phase-decoder-' + str(k), phiEncoderModel.layers[pCount].get_weights())
        pdc += 1

for mCount in xrange(len(magEncoderModel.layers)):
    if mCount in magEncIndex:
        np.save('./lstmae-weights/lstmae-magnitude-encoder-' + str(mec), magEncoderModel.layers[mCount].get_weights())
        mec += 1
    elif mCount in magDecIndex:
        k = (len(magDecIndex) - 1) - mdc
        np.save('./lstmae-weights/lstmae-magnitude-decoder-' + str(k), magEncoderModel.layers[mCount].get_weights())
        mdc += 1
