import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from LBG import lbg
from MFCC import MFCC_Coeff
from LinearPredictionCoefficients import lpc
import numpy as np



def training(nfiltbank, orderLPC):
    trainingSet = 4
    nCentroid = 16
    cbMfcc = np.empty((trainingSet,nfiltbank,nCentroid))
    cbLpc = np.empty((trainingSet, orderLPC, nCentroid))
    directory = os.getcwd() + '/train';
    fname = str()

    for i in range(trainingSet):
        fname = '/s' + str(i+1) + '.wav'
        print 'Voice ', str(i+1), 'is trained' 
        (fs,s) = read(directory + fname)
        MFCC = MFCC_Coeff(s, fs, nfiltbank)
        lpc_coeff = lpc(s, fs, orderLPC)
        cbMfcc[i,:,:] = lbg(MFCC, nCentroid)
        cbLpc[i,:,:] = lbg(lpc_coeff, nCentroid)
        
        plt.figure(i)
        plt.title('Codebook for speaker ' + str(i+1) + ' with ' + str(nCentroid) +  ' centroids')
        for j in range(nCentroid):
            plt.subplot(211)
            plt.stem(cbMfcc[i,:,j])
            plt.ylabel('MFCC')
            plt.subplot(212)
            markerline, stemlines, baseline = plt.stem(cbLpc[i,:,j])
            plt.setp(markerline,'markerfacecolor','r')
            plt.setp(baseline,'color', 'k')
            plt.ylabel('LPC')
            plt.axis(ymin = -1, ymax = 1)
            plt.xlabel('Number of features')
    plt.show()
    print 'Training has been performed '
    codebooks = np.empty((2, nfiltbank, nCentroid))
    MFCC = np.empty((2, nfiltbank, 68))
   
    for i in range(2):
        fname = '/s' + str(i+2) + '.wav'
        (fs,s) = read(directory + fname)
        MFCC[i,:,:] = MFCC_Coeff(s, fs, nfiltbank)[:,0:68]
        codebooks[i,:,:] = lbg(MFCC[i,:,:], nCentroid)    
    plt.figure(trainingSet + 1)
    s1 = plt.scatter(MFCC[0,6,:], MFCC[0,4,:],s = 100,  color = 'r', marker = 'o')
    c1 = plt.scatter(codebooks[0,6,:], codebooks[0,4,:], s = 100, color = 'r', marker = '+')
    s2 = plt.scatter(MFCC[1,6,:], MFCC[1,4,:],s = 100,  color = 'b', marker = 'o')
    c2 = plt.scatter(codebooks[1,6,:], codebooks[1,4,:], s = 100, color = 'b', marker = '+')
    plt.grid()
    plt.legend((s1, s2, c1, c2), ('Sp1','Sp2','Sp1 centroids', 'Sp2 centroids'), scatterpoints = 1, loc = 'upper left')    
    plt.show()
   
    
    return (cbMfcc, cbLpc)
    
    
