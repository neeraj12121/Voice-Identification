import numpy as np

def EucledianDistance(d,c):
    n = np.shape(d)[1]
    p = np.shape(c)[1]
    dist = np.empty((n,p))
    if n<p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:,i], (p,1)))
            dist[i,:] = np.sum((copies - c)**2,0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:,i],(n,1)))
            dist[:,i] = np.transpose(np.sum((d - copies)**2,0))   
    dist = np.sqrt((dist))
    return dist
            

def lbg(f, M):
    eps = 0.01
    CB = np.mean(f, 1)       #codebook
    distortion = 1
    nCentroid = 1
    while nCentroid < M:
        new_CB = np.empty((len(CB), nCentroid*2))
        if nCentroid == 1:
            new_CB[:,0] = CB*(1+eps)
            new_CB[:,1] = CB*(1-eps)
        else:    
            for i in range(nCentroid):
                new_CB[:,2*i] = CB[:,i] * (1+eps)
                new_CB[:,2*i+1] = CB[:,i] * (1-eps)
        
        CB = new_CB
        nCentroid = np.shape(CB)[1]
        D = EucledianDistance(f, CB)
        
        
        while np.abs(distortion) > eps:
            prev_distance = np.mean(D)
            nearest_CB = np.argmin(D,axis = 1)       
            for i in range(nCentroid):
                CB[:,i] = np.mean(f[:,np.where(nearest_CB == i)], 2).T
            CB = np.nan_to_num(CB)   
            D = EucledianDistance(f, CB)
            distortion = (prev_distance - np.mean(D))/prev_distance
    return CB
        
            
