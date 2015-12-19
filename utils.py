import wave, struct, time
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft

def openWavFile(fileName):
    data = wav.read(fileName)
    ssize = data[1].shape[0]
    nparray = data[1].astype('float32')
    return nparray

def stftWindowFunction(xPhi, xMag):
    oldShapePhi = xPhi.shape
    oldShapeMag = xMag.shape
    xPhi = np.reshape(xPhi, (-1, xPhi.shape[-1]))
    xMag = np.reshape(xMag, (-1, xMag.shape[-1]))
    retValPhi = []
    retValMag = []
    for xValPhi, xValMag in zip(xPhi, xMag):
        w = np.hanning(xValPhi.shape[0])
        phiObj = np.zeros(xValPhi.shape[0], dtype=complex)
        phiObj.real, phiObj.imag = np.cos(xValPhi), np.sin(xValPhi)
        xIfft = np.fft.ifft(xValMag * phiObj)
        wFft = np.fft.fft(w*xIfft.real)
        retValPhi.append(np.angle(wFft))
        retValMag.append(np.abs(wFft))
    retValMag = np.reshape(retValMag, oldShapeMag)
    retValPhi = np.reshape(retValPhi, oldShapePhi)
    return retValPhi, retValMag

def stft(x, framesz, hop):
    framesamp = int(framesz)
    hopsamp = int(hop)
    #w = np.hanning(framesamp)
    X = np.asarray([np.fft.fft(x[i:i+framesamp]) for i in range(0, len(x) - framesamp, hopsamp)])
    xPhi = np.angle(X)
    xMag = np.abs(X)
    return xPhi, xMag

def istft(X, fs, hop, origs):
    x = np.zeros(origs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += np.real(np.fft.ifft(X[n]))
    return x

def waveToSTFT(waveData, sampCount, blkSize, hop):
    initLen = len(waveData)
    sampSize = int(initLen/sampCount)
    
    phiObj = []
    magObj = []
    for sInd in xrange(0, sampCount):
        tempTmSpls = []
        sampBlk = waveData[sInd * sampSize:(sInd + 1) * sampSize]
        stftPhi, stftMag = stft(sampBlk, blkSize, hop)
        phiObj.append(stftPhi)
        magObj.append(stftMag)
    return ([], np.asarray(phiObj), np.asarray(magObj))
    
def waveToMFCC(waveData, sampCount, blkCount=False, blkSize=False):
    waveLen = len(waveData)
    sampSize = int(waveLen/sampCount)
    retTmSpl = []
    if blkSize:
        blkCount = sampSize/blkSize
    elif blkCount:
        blkSize = sampSize/blkCount
    else:
        return False
    for sInd in xrange(0, sampCount):
        tempTmSpls = []
        sampBlk = waveData[sInd * sampSize:(sInd + 1) * sampSize]
        for bInd in xrange(0, blkCount):
            tempBlk = sampBlk[bInd * blkSize:(bInd + 1) * blkSize]
            
            complexSpectrum = np.fft.fft(tempBlk)
            powerSpectrum = np.abs(complexSpectrum) ** 2
            filteredSpectrum = powerSpectrum
            logSpectrum = np.log(filteredSpectrum)
            dctSpectrum = fft.dct(logSpectrum, type=2)
            tempTmSpls.append(dctSpectrum)
        retTmSpl.append(tempTmSpls)
    retTmSpl = np.asarray(retTmSpl)
    return retTmSpl

def waveToBlock(waveData, sampCount, blkCount=False, blkSize=False, olapf=1, shift=False):
    if shift:
        waveData = np.concatenate((waveData[shift:], waveData[:shift]))
    waveLen = len(waveData)
    sampSize = int(waveLen/sampCount)
    retPhase = []
    retMag = []
    retTmSpl = []
    if blkSize and blkCount:
        tlen = sampCount * blkCount * blkSize
        sampSize = blkCount * blkSize
        diff = tlen - waveLen
        if diff > 0:
            waveData = np.pad(waveData, (0,diff), 'constant', constant_values=0)
    elif blkSize:
        blkCount = sampSize/blkSize
    elif blkCount:
        blkSize = sampSize/blkCount
    else:
        return False
    for sInd in xrange(0, sampCount):
        tempPhases = []
        tempMags = []
        tempTmSpls = []
        sampBlk = waveData[sInd * sampSize:(sInd + 1) * sampSize]
        for bInd in xrange(0, blkCount - (olapf - 1)):
            tempBlk = sampBlk[bInd * blkSize:(bInd + olapf) * blkSize]
            tempFFT = np.fft.fft(tempBlk)
            tempPhase = np.angle(tempFFT)
            tempMagn = np.abs(tempFFT)
            tempPhases.append(tempPhase)
            tempMags.append(tempMagn)
            tempTmSpls.append(tempBlk)
        retPhase.append(tempPhases)
        retMag.append(tempMags)
        retTmSpl.append(tempTmSpls)
    retPhase = np.asarray(retPhase)
    retTmSpl = np.asarray(retTmSpl)
    retMag = np.asarray(retMag)
    return (retTmSpl, retPhase, retMag)

def sectionFeatureScaling(data):
    dataShape = data.shape
    flatData = np.copy(data).flatten()
    flatMax = np.max(flatData)
    flatMin = np.min(flatData)
    scaledData = (flatData - flatMin)/(flatMax- flatMin)
    scaledData = np.reshape(scaledData, dataShape)
    return scaledData, flatMax, flatMin

def blockFeatureScaling(kData):
    data = np.copy(kData)
    maxVal = np.max(np.max(data, axis=0), axis=0)
    minVal = np.min(np.min(data, axis=0), axis=0)
    scaledData = (data - minVal)/(maxVal- minVal)
    return scaledData, maxVal, minVal
    
def sectionNormalize(data):
    dataShape = data.shape
    flatData = np.copy(data).flatten()
    flatMean = np.mean(flatData)
    flatStd = np.std(flatData)
    scaledData = (flatData - flatMean)/flatStd
    scaledData = np.reshape(scaledData, dataShape)
    return scaledData, flatMean, flatStd

def blockNormalize(data):
    dataStartShape = data.shape
    if len(data.shape) == 2:
        data = np.reshape(data, (1, data.shape[0], data.shape[1]))
    
    if len(data.shape) == 1:
        data = np.reshape(data, (1, 1, data.shape[0]))
    npNorm = np.zeros_like(data)
    xCount = data.shape[0]
    yCount = data.shape[1]
    for sectInd in xrange(xCount):
        for blockInd in xrange(yCount):
            npNorm[sectInd][blockInd] = data[sectInd][blockInd]
    mean = np.mean(np.mean(npNorm, axis=0), axis=0)
    std = np.sqrt(np.mean(np.mean(np.abs(npNorm-mean)**2, axis=0), axis=0))
    std = np.maximum(1.0e-8, std)
    norm = npNorm.copy()
    norm[:] -= mean
    norm[:] /= std
    return norm, mean, std
    
def extractSTFTWaveData(wavData, sampCount, blkSize=False, returnObj="all", olapf=100):#(waveData, sampCount, blkCount, blkSize, hop):
    #wavData = openWavFile(fileName)
    #wavObj, wavPhi, wavMag = waveToBlock(wavData, sampCount, blkCount=blkCount, blkSize=blkSize, olapf=olapf, shift=False)
    wavObj, wavPhi, wavMag = waveToSTFT(wavData, sampCount, blkSize=blkSize, hop=olapf)
    #mfccObj = waveToMFCC(wavData, sampCount, blkCount, blkSize)
    phiWav, meanPhiWav, stdPhiWav = blockNormalize(wavPhi)
    magWav, meanMagWav, stdMagWav = blockNormalize(wavMag)
    #MfccWav, meanMfcc, stdMfcc = blockNormalize(mfccObj)
    
    if returnObj == "phase":
        return phiWav, meanPhiWav, stdPhiWav
    elif returnObj == "magnitude":
        return magWav, meanMagWav, stdMagWav
    else:
        return phiWav, meanPhiWav, stdPhiWav, magWav, maxMagWav, minMagWav
        
def extractWaveData(wavData, sampCount, blkCount=False, blkSize=False, returnObj="all", olapf=1, shift=False):
    #wavData = openWavFile(fileName)
    wavObj, wavPhi, wavMag = waveToBlock(wavData, sampCount, blkCount=blkCount, blkSize=blkSize, olapf=olapf, shift=False)
    #mfccObj = waveToMFCC(wavData, sampCount, blkCount, blkSize)
    phiWav, meanPhiWav, stdPhiWav = blockNormalize(wavPhi)
    magWav, meanMagWav, stdMagWav = blockNormalize(wavMag)
    #MfccWav, meanMfcc, stdMfcc = blockNormalize(mfccObj)
    
    if returnObj == "phase":
        return phiWav, meanPhiWav, stdPhiWav
    elif returnObj == "magnitude":
        return magWav, meanMagWav, stdMagWav
    else:
        return phiWav, meanPhiWav, stdPhiWav, magWav, maxMagWav, minMagWav
        
def blockShift(data, shift=1):
    retObj = []
    for sectInd in xrange(data.shape[0]):
        retObj.append( np.concatenate((data[sectInd][shift:], data[sectInd][0:shift])) )
    return np.reshape(retObj, data.shape)