from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct
import numpy as np
import matplotlib.pyplot as plt

def h2m(freq):
    return 1125*np.log(1 + freq/700)
    
def m2h(m):
    return 700*(np.exp(m/1125) - 1)
 
def filter(nfft, nfiltbank, fs):
    lmel = h2m(300)
    umel = h2m(8000)
    mel = np.linspace(lmel, umel, nfiltbank+2)
    hertz = [m2h(m) for m in mel]
    fbins = [int(hz * (nfft/2+1)/fs) for hz in hertz]
    fbank = np.empty((int(nfft/2+1),nfiltbank))
    for i in range(1,nfiltbank+1):
        for k in range(int(nfft/2 + 1)):
            if k < fbins[i-1]:
                fbank[k, i-1] = 0
            elif k >= fbins[i-1] and k < fbins[i]:
                fbank[k,i-1] = (k - fbins[i-1])/(fbins[i] - fbins[i-1])
            elif k >= fbins[i] and k <= fbins[i+1]:
                fbank[k,i-1] = (fbins[i+1] - k)/(fbins[i+1] - fbins[i])
            else:
                fbank[k,i-1] = 0
    return fbank
            
def MFCC_Coeff(s,fs, nfiltbank):
    numberSamples = np.int32(0.025*fs)
    total = np.int32(0.01*fs)
    numberFrames = np.int32(np.ceil(len(s)/(numberSamples-total)))
    padding = ((numberSamples-total)*numberFrames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((numberSamples, numberFrames))
    start = 0
    for i in range(numberFrames):
        segment[:,i] = signal[start:start+numberSamples]
        start = (numberSamples-total)*i
    nfft = 512
    periodogram = np.empty((int(numberFrames),int(nfft/2 + 1)))
    for i in range(numberFrames):
        x = segment[:,i] * hamming(numberSamples)
        spectrum = fftshift(fft(x,nfft))
        periodogram[i,:] = abs(spectrum[int(nfft/2-1):])/numberSamples
    fbank = filter(nfft, nfiltbank, fs)
    mfcc = np.empty((nfiltbank,numberFrames))
    for i in range(nfiltbank):
        for k in range(numberFrames):
            mfcc[i,k] = np.sum(periodogram[k,:]*fbank[:,i])
            
    mfcc = np.log10(mfcc)
    mfcc = dct(mfcc)
    mfcc[0,:]= np.zeros(numberFrames)
    return mfcc
            

