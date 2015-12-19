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

mx = magntData.shape[0]
my = magntData.shape[1]
mz = magntData.shape[2]

kMagnData = np.load('./coded-data/' + settings['coded-file'] + '-phase.npy')
yMagnConcat = blockShift(kMagnData, shift=1)

kPhaseData = np.load('./coded-data/' + settings['coded-file'] + '-magnitude.npy')
yConcat = blockShift(kPhaseData, shift=1)
print kMagnData.shape
yx = kMagnData.shape[0]
yy = kMagnData.shape[1]
yz = kMagnData.shape[2]

ox = kPhaseData.shape[0]
oy = kPhaseData.shape[1]
oz = kPhaseData.shape[2]

model = Sequential()
model.add(TimeDistributedDense(input_dim=yz, output_dim=yz + 120))
model.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
model.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
model.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
model.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
model.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
model.add(TimeDistributedDense(input_dim=yz + 120, output_dim=oz))
model.load_weights('./lstm-weights/phi-' + settings['lstm-file'])
model.compile(loss='mean_squared_error', optimizer='rmsprop')

magnModel = Sequential()
magnModel.add(TimeDistributedDense(input_dim=yz, output_dim=yz + 120))
magnModel.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
magnModel.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
magnModel.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
magnModel.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
magnModel.add(LSTM(input_dim=yz + 120, output_dim=yz + 120, return_sequences=True))
magnModel.add(TimeDistributedDense(input_dim=yz + 120, output_dim=oz))
magnModel.load_weights('./lstm-weights/mag-' + settings['lstm-file'])
magnModel.compile(loss='mean_squared_error', optimizer='rmsprop')


savePhase = []
saveMagnd = []
for sectIndex in xrange(1):#settings['section-count']):
    basePhase = np.copy(kPhaseData[sectIndex:sectIndex+1,0:-1])
    baseMagnd = np.copy(kMagnData[sectIndex:sectIndex+1,0:-1])
    for j in range(0, 20):
        basePhaseR = model.predict(basePhase)
        baseMagndR = magnModel.predict(baseMagnd)
        savePhase.append(basePhaseR[0,-1,:])
        saveMagnd.append(baseMagndR[0,-1,:])
        basePhase = np.concatenate((basePhase[0,1:], [basePhaseR[0,-1,:]])).reshape(1,-1,basePhase.shape[2])
        baseMagnd = np.concatenate((baseMagnd[0,1:], [baseMagndR[0,-1,:]])).reshape(1,-1,baseMagnd.shape[2])
        print '----------' + str(j) + ' (' + str(np.mean(basePhaseR[0,-1,:])) + ', ' + str(np.std(basePhaseR[0,-1,:])) + ')' + '----------'

np.save('./coded-data/' + settings['coded-file'] + '-phase-result', savePhase)
np.save('./coded-data/' + settings['coded-file'] + '-magnitude-result', saveMagnd)
