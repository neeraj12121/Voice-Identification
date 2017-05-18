import numpy as np
from scipy.io.wavfile import read
from LBG import EucledianDistance
from MFCC import MFCC_Coeff
from LinearPredictionCoefficients import lpc
from Train import training
import os


nCorrect_MFCC = 0
nCorrect_LPC = 0
trainingSet = 4
q1 = 12
q2 = 15
(cbMfcc, cbLpc) = training(q1, q2)
directory = os.getcwd() + '/test';
fname = str()



def minDistance(f, c):
    person = 0
    minDist = np.inf
    for k in range(np.shape(c)[0]):
        D = EucledianDistance(f, c[k,:,:])
        dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0]) 
        if dist < minDist:
            minDist = dist
            person = k
    return person
    

for i in range(nPerson):
    fname = '/s' + str(i+1) + '.wav'
    print 'Now Voice ', str(i+1), 'f tested'
    (fs,s) = read(directory + fname)
    mel_coefs = MFCC_Coeff(s,fs,q1)
    lpc_coefs = lpc(s, fs, q2)
    sp_mfcc = minDistance(mel_coefs, cbMfcc)
    sp_lpc = minDistance(lpc_coefs, cbLpc)
    print 'Voice ', (i+1), ' in test matches with Voice ', (sp_mfcc+1), ' in train for training with MFCC'
    print 'Voice ', (i+1), ' in test matches with Voice ', (sp_lpc+1), ' in train for training with LPC'
    if i == sp_mfcc:
        nCorrect_MFCC += 1
    if i == sp_lpc:
        nCorrect_LPC += 1
percentageCorrect_MFCC = (nCorrect_MFCC/nPerson)*100
print 'Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%'
percentageCorrect_LPC = (nCorrect_LPC/nPerson)*100
print 'Accuracy of result for training with LPC is ', percentageCorrect_LPC, '%'


    
