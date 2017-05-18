import numpy as np
import matplotlib.pyplot as plt

def autoCorrect(x):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
  
def SymMat(acf,p):
    R = np.empty((p,p))
    for i in range(p):
        for j in range(p):
            R[i,j] = acf[np.abs(i-j)]
    return R
    
def lpc(s,fs,p):
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
    LPC = np.empty((p, numberFrames))
    for i in range(numberFrames):
        acf = autoCorrect(segment[:,i])
        r = -acf[1:p+1].T
        R = SymMat(acf,p)
        LPC[:,i] = np.dot(np.linalg.inv(R),r)
        LPC[:,i] = LPC[:,i]/np.max(np.abs(LPC[:,i]))
             
    return LPC
