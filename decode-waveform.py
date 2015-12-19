import wave, struct, time, random, math
import numpy as np
import os
import scipy.io.wavfile as wav
import config as cfg
import datetime

settings = cfg.getConfig()
rate, wavSrc = wav.read(settings['source'])

phiDist = np.load('./results-data/' + settings['phase-result'] + 'phase-data.npy')
phiSize = phiDist.shape[2]
#print phiDist.shape
phiDist = np.reshape(phiDist, (phiDist.shape[0] * phiDist.shape[1], phiDist.shape[2]))
phiMean = np.mean(phiDist, axis=0)
phiStd = np.std(phiDist, axis=0)
phiNorm = (phiDist - phiMean) / phiStd

magDist = np.load('./results-data/' + settings['magnitude-result'] + 'magnitude-data.npy')
magDist = np.reshape(magDist, (magDist.shape[0] * magDist.shape[1], magDist.shape[2]))
magMean = np.mean(magDist, axis=0)
magStd = np.std(magDist, axis=0)
magNorm = (magDist - magMean) / magStd

phiShape = phiDist.shape[1]
phiCount = phiDist.shape[0]
validLen = int(wavSrc.shape[0]/phiShape) * phiShape
splitWav = np.asarray(np.split(wavSrc[:validLen], int(wavSrc.shape[0]/phiShape)))

lambFtt = lambda data: np.fft.fft(data)
magFft = lambda data: np.abs(data)
phiFft = lambda data: np.angle(data)
fftData = [lambFtt(wavData) for wavData in splitWav]
wavMags = [magFft(fftDatum) for fftDatum in fftData]
wavPhis = [phiFft(fftDatum) for fftDatum in fftData]

wmMean = np.mean(wavMags, axis=0)
wmStd = np.std(wavMags, axis=0)
wpMean = np.mean(wavPhis, axis=0)
wpStd = np.std(wavPhis, axis=0)

phiRenorm = (phiNorm * wpStd) + wpMean
magRenorm = (magNorm * wmStd) + wmMean
magRenorm = (magNorm * wmStd) + wmMean

wavOut = []

for phiVal, magVal in zip(phiRenorm, magRenorm):
    jonHamm = np.hamming(magVal.shape[0])#np.kaiser(magVal.shape[0], 10)
    phiOut = np.zeros(phiVal.shape, dtype=complex)
    phiOut.real, phiOut.imag = np.cos(phiVal), np.sin(phiVal)
    myIfft = jonHamm * np.fft.ifft(magVal * phiOut)
    wavOut.append(myIfft)
n = np.zeros(len(wavOut)*phiSize)
kmax = np.max(wavOut)

k = 0
j = phiSize
blockSize = phiSize / 2
for wavData in wavOut:
    n[k:k+j] += wavData.astype('int16') / (32767.0)
    k += blockSize
Xnew = n#wavOut.astype('int16')
dateStr = str(datetime.datetime.now().date()) + '-' + str(datetime.datetime.now().time().hour) + '-' + str(datetime.datetime.now().time().minute)
wav.write('./results-wav/' + dateStr + '.wav', settings['sample-rate'], Xnew)
