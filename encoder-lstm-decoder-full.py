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
        phiEncoderModel.add(AutoEncoderDropout(0.4))
    phiEncoderModel.add(timeDistDense)
    phiEncoderModel.add(ELU())
    last = pz - outlay

phiEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
phiEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
phiEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
phiEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))

magEncoderModel = Sequential()    
for encInd in range(0, layerCount):
    midlay = (encInd) * settings['dim-decrease']
    outlay = (encInd + 1) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, input_length=py)
    if encInd > 0:
        magEncoderModel.add(AutoEncoderDropout(0.4))
    magEncoderModel.add(timeDistDense)
    magEncoderModel.add(ELU())
    last = pz - outlay

magEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
magEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
magEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))
magEncoderModel.add(LSTM(input_dim=last, output_dim=last, return_sequences=True))

print len(magEncoderModel.layers)
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear')
    phiEncoderModel.add(timeDistDense)

phiEncoderModel = modelWeightsLoader(phiEncoderModel, './lstm-weights/phi-' + settings['lstm-file'], {0:11, 1:12})
phiEncoderModel = modelWeightsLoader(phiEncoderModel, './autoencoder-weights/' + settings['phase-encoder'] + '-phase-AE', {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:13,12:14,13:15,14:16,15:17,16:18})
phiEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')
  
for encIndM in range(0, layerCount):
    encInd = (layerCount - 1) - encIndM
    midlay = (encInd + 1) * settings['dim-decrease']
    outlay = (encInd) * settings['dim-decrease']
    timeDistDense = TimeDistributedDense(input_dim=pz - midlay, output_dim=pz - outlay, activation='linear')
    magEncoderModel.add(timeDistDense)

magEncoderModel = modelWeightsLoader(magEncoderModel, './lstm-weights/mag-' + settings['lstm-file'], {0:11, 1:12})
magEncoderModel = modelWeightsLoader(magEncoderModel, './autoencoder-weights/' + settings['magnitude-encoder'] + '-magnitude-AE', {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:13,12:14,13:15,14:16,15:17,16:18})
magEncoderModel.compile(loss='mean_squared_error', optimizer='rmsprop')

print 'start'
magResult = []
phiResult = []
for k in range(3):
    magPredict = np.copy(magntData[k:k+1,150:])
    phiPredict = np.copy(phaseData[k:k+1,150:])
    for indVal in range(688):
        magPredRes = magEncoderModel.predict(magPredict)
        phiPredRes = phiEncoderModel.predict(phiPredict)
        magPredict = np.append(magPredict[0][1:], [magPredRes[0][-1]]).reshape(1, magPredict.shape[1], -1)
        phiPredict = np.append(phiPredict[0][1:], [phiPredRes[0][-1]]).reshape(1, phiPredict.shape[1], -1)
        magResult.append(magPredRes[0][-1])
        phiResult.append(phiPredRes[0][-1])

magResult = np.asarray(magResult)
magResult = np.reshape(magResult, (1, magResult.shape[0], magResult.shape[1]))

phiResult = np.asarray(phiResult)
phiResult = np.reshape(phiResult, (1, phiResult.shape[0], phiResult.shape[1]))

np.save('./results-data/' + settings['phase-result'] + 'phase-data', phiResult)
np.save('./results-data/' + settings['magnitude-result'] + 'magnitude-data', magResult)
