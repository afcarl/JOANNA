from keras.layers.core import TimeDistributedDense, Activation, AutoEncoderDropout
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.models import Sequential

def buildPrestackedAutoencoder(layerCount, dimDecrease, input_dim, dropout=0.1, encActivation='softplus', decActivation='linear'):
    AEModel = Sequential()
    phiEncIndex = []
    phiDecIndex = []
    for encInd in range(0, layerCount):
        midlay = (encInd) * dimDecrease
        outlay = (encInd + 1) * dimDecrease
        timeDistDense = TimeDistributedDense(input_dim=input_dim - midlay, output_dim=input_dim - outlay, activation=encActivation)
        if encInd > 0 and dropout:
            layCount = len(AEModel.layers)
            AEModel.add(AutoEncoderDropout(dropout))
        layCount = len(AEModel.layers)
        phiEncIndex.append(layCount)
        AEModel.add(timeDistDense)

    for encIndM in range(0, layerCount):
        encInd = (layerCount - 1) - encIndM
        midlay = (encInd + 1) * dimDecrease
        outlay = (encInd) * dimDecrease
        timeDistDense = TimeDistributedDense(input_dim=input_dim - midlay, output_dim=input_dim - outlay, activation=decActivation)
        layCount = len(AEModel.layers)
        phiDecIndex.append(layCount)
        AEModel.add(timeDistDense)
    return AEModel, phiEncIndex, phiDecIndex
    
def buildLSTMModel(layerCount, input_dim, added_dim):
    model = Sequential()
    model.add(TimeDistributedDense(input_dim=input_dim, output_dim=input_dim + added_dim))
    for lcount in range(layerCount):
        model.add(LSTM(input_dim=input_dim + added_dim, output_dim=input_dim + added_dim, return_sequences=True))
    model.add(TimeDistributedDense(input_dim=input_dim + added_dim, output_dim=input_dim))
    return model