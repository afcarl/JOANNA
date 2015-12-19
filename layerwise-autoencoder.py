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
import wave, struct, time
import numpy as np
import os
import scipy.io.wavfile as wav
import scipy.fftpack as fft
    
audioData = openWavFile('4-samples-sapo.wav')
phaseData, meanPhiWav, stdPhiWav = extractWaveData(audioData, 8, blkCount=800, returnObj="phase", olapf=4)
magntData, maxMagWav, minMagWav = extractWaveData(audioData, 8, blkCount=800, returnObj="magnitude", olapf=4)
#mfccData, meanMfcc, stdMfcc = extractWaveData(audioData, 2, blkCount=200, returnObj="mfcc")

px = phaseData.shape[0]
py = phaseData.shape[1]
pz = phaseData.shape[2]

mx = magntData.shape[0]
my = magntData.shape[1]
mz = magntData.shape[2]

print phaseData.shape
print magntData.shape

kMagnData = np.copy(magntData)
for encInd in range(0, 5):
    encMag = Sequential()
    midlay = encInd * 200
    outlay = (encInd + 1) * 200
    encoder = containers.Sequential([TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='tanh')])
    decoder = containers.Sequential([TimeDistributedDense(input_dim=pz - outlay, output_dim=pz - midlay, activation='linear')])
    encMag.add(Dropout(p=0.1, input_shape=(py, pz - midlay)))
    encMag.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    #encMag.load_weights('magnitude-layer-' + str(encInd))
    encMag.compile(loss='mean_squared_error', optimizer='rmsprop')
    startLoss = 999
    losses = []
    times = []
    while len(losses) < 20:
        start = time.time()
        rnum = np.random.randint(0,4)
        myBatch = 1
        encMag.fit(kMagnData, kMagnData, batch_size=myBatch, nb_epoch=1000, verbose=0, validation_split=0.0, shuffle=False)
        startLoss = encMag.train_on_batch(kMagnData, kMagnData)
        losses.append(startLoss)
        end = time.time()
        timeDiff = end - start
        times.append(timeDiff)
        print 'magnitude autoencoder-layer ' + str(encInd) + ', epoch: ' + str(len(losses)) + ', loss: ' + str(startLoss)
        print 'time: ' + str(times)
        times = []
    encMag.save_weights('magnitude-layer-' + str(encInd), overwrite=True)
    np.save('./olap-aeparams/drop-magnitude-encoder-layer-' + str(encInd), encMag.layers[1].encoder.get_weights())
    np.save('./olap-aeparams/drop-magnitude-decoder-layer-' + str(encInd), encMag.layers[1].decoder.get_weights())
    kMagnData = encMag.predict(kMagnData)

kPhaseData = np.copy(phaseData)
for encInd in range(0, 5):
    encPhi = Sequential()
    midlay = encInd * 200
    outlay = (encInd + 1) * 200
    encoder = containers.Sequential([TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='tanh')])
    decoder = containers.Sequential([TimeDistributedDense(input_dim=pz - outlay, output_dim=pz - midlay, activation='linear')])
    encPhi.add(Dropout(p=0.1, input_shape=(py, pz - midlay)))
    encPhi.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    #encPhi.load_weights('phase-layer-' + str(encInd))
    encPhi.compile(loss='mean_squared_error', optimizer='rmsprop')
    startLoss = 999
    losses = []
    times = []
    while len(losses) < 20:
        start = time.time()
        rnum = np.random.randint(0,4)
        myBatch = 1
        encPhi.fit(kPhaseData, kPhaseData, batch_size=myBatch, nb_epoch=1000, verbose=0, validation_split=0.0, shuffle=False)
        startLoss = encPhi.train_on_batch(kPhaseData, kPhaseData)
        losses.append(startLoss)
        end = time.time()
        timeDiff = end - start
        times.append(timeDiff)
        print 'phase autoencoder-layer ' + str(encInd) + ', epoch: ' + str(len(losses)) + ', loss: ' + str(startLoss)
        print 'time: ' + str(times)
        times = []
    encPhi.save_weights('phase-layer-' + str(encInd), overwrite=True)
    np.save('./olap-aeparams/drop-phase-encoder-layer-' + str(encInd), encPhi.layers[1].encoder.get_weights())
    np.save('./olap-aeparams/drop-phase-decoder-layer-' + str(encInd), encPhi.layers[1].decoder.get_weights())
    kPhaseData = encPhi.predict(kPhaseData)
