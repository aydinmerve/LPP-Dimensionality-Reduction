# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:20:29 2019

"""
import numpy as np
import random as rand
import numpy.matlib as nm
import math
from scipy.sparse import coo_matrix, isspmatrix, spdiags
from scipy.sparse.linalg import  eigsh, ArpackNoConvergence, eigs


def KGE(W, D, options, data):
    
    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1
    
    if 'reducedDim' in options:
        reducedDim = options['reducedDim']
    else:
        reducedDim = 30
    if 'Regu' not in options:
        bPCA = 1
    else:
        bPCA = 0
        if 'ReguAlpha' not in options:
            options['ReguAlpha'] = 0.01
    bD = 1
    if len(D.toarray())==0:
        bD = 0
    if 'Kernel' in options:
        K = data
    else:
        K = constructKernelKLPP(data,[],options)
    
    nSmp = K.shape[0]
    if W.shape[0] != nSmp:
        print('W and data mismatch!')
    if bD==1 and D.shape[0] != nSmp:
        print('D and data mismatch!')
    
    
    sumK = np.sum(K,axis=1)
    shape = np.reshape(sumK/nSmp,(nSmp,1))
    H = nm.repmat(shape,1,nSmp)
    K = K - H - H.transpose()
    K = np.add(K,np.sum(sumK)/np.power(nSmp,2))
    K = np.maximum(K,K.transpose())
    
    if bPCA == 1:
        eigvalue_PCA, eigvector_PCA = np.linalg.eig(K)
        index = np.argsort(eigvalue_PCA)
        eigvalue_PCA = eigvalue_PCA[index]
        eigvector_PCA = eigvector_PCA[:,index]
        maxEigValue = max(np.abs(eigvalue_PCA))
        eigIdx = np.where(np.abs(eigvalue_PCA)/maxEigValue > 1e-6)[0]
        if len(eigIdx) == len(eigvalue_PCA):
            idx = np.argmin(eigvalue_PCA)
            eigIdx = list(range(len(eigvalue_PCA)))
            del eigIdx[idx]
        eigvalue_PCA = eigvalue_PCA[eigIdx]
        eigvector_PCA = eigvector_PCA[:,eigIdx]
        
        K = eigvector_PCA
        if bD == 1:
            DPrime = np.dot(np.dot(K,D),K)
            DPrime = np.maximum(DPrime,DPrime.transpose())
    else:
        if bD == 1:
            DPrime = np.dot(np.dot(K,D.toarray()),K)
        else:
            np.dot(K,K)
        for i in range(len(DPrime)):
            DPrime[i,i] = DPrime[i,i] + options['ReguAlpha']
        DPrime = np.maximum(DPrime,DPrime.transpose())
    
    if isinstance(W, (np.ndarray)):
        WPrime = np.dot(np.dot(K,W),K)
    else:
        WPrime = np.dot(np.dot(K,W.toarray()),K)
    WPrime = np.maximum(WPrime,WPrime.transpose())
    
    dimMatrix = WPrime.shape[1]
    if reducedDim > dimMatrix:
        reducedDim = dimMatrix
    if 'bEigs' in options:
        bEigs = options['bEigs']
    else:
        if dimMatrix > MAX_MATRIX_SIZE and reducedDim < dimMatrix*EIGVECTOR_RATIO:
            bEigs = 1
        else:
            bEigs = 0
    
    if bEigs == 1:
        if bPCA == 1 and bD ==0:
            eigvalue, eigvector = eigsh(WPrime, k=reducedDim, which='LA')
        else:
            try:
                eigvalue, eigvector = eigsh(WPrime, k=reducedDim, M=DPrime, which='LA', maxiter=40)
            except ArpackNoConvergence as e:
                    print(e)
                    eigvalue = e.eigenvalues
                    eigvector = e.eigenvectors
                    print(eigvalue.shape, eigvector.shape)
    else:
        if bPCA == 1 and bD ==0:
            eigvalue, eigvector = eigsh(WPrime, k = len(WPrime)-1)
        else:
            eigvalue, eigvector = eigsh(WPrime, k = len(WPrime)-1, M=DPrime)
        index = np.argsort(-eigvalue)
        eigvalue = eigvalue[index]
        eigvector = eigvector[:,index]
        
        if reducedDim < eigvector.shape[1]:
            eigvector = eigvector[:,0:reducedDim]
            eigvalue = eigvalue[0:reducedDim]
    
    if bPCA == 1:
        eigvalue_PCA = np.power(eigvalue_PCA,-1)
        eigvector = K*(nm.repmat(eigvalue_PCA,len(eigvalue),1)*eigvector)
    
    tmpNorm = np.sqrt(np.sum(np.dot(eigvector.transpose(),K)*eigvector.transpose(),axis=1))
    shape = np.reshape(tmpNorm,(len(tmpNorm),1))
    eigvector = eigvector / nm.repmat(shape,1,len(eigvector)).transpose()    
    
    return eigvector, eigvalue

def KLPP(W, options, data):
    
    if 'Kernel' in options:
        K = data
    else:
        K = constructKernelKLPP(data,[],options)
        nSmp = K.shape[0]
        D = np.sum(W,axis=1)
        if 'Regu' in options:
            options['ReguAlpha'] = options['ReguAlpha']*np.sum(D)/len(D)
        D = coo_matrix((np.array(D).ravel(),(range(nSmp),range(nSmp))),shape=(nSmp,nSmp))
        options['Kernel'] = 1
        eigvector, eigvalue = KGE(W, D, options, K)
        
        eigIdx = np.where(eigvalue > 1e-3)[0]
        eigvalue = eigvalue[eigIdx]
        eigvector = eigvector[:,eigIdx]
        return eigvector, eigvalue
        

def EuDist2(fea_a,fea_b,bSqrt):
    if(len(fea_b)==0):
        squareFeaA = fea_a*fea_a#Eleman eleman çarpimi
        sumFeaAA = np.sum(squareFeaA,axis=1) ## satir elemanlarini toplar
        sumFeaAB = np.dot(fea_a,fea_a.transpose())#matris carpimi
        
        if(isspmatrix(np.array(sumFeaAA))):
            sumFeaAA.toarray()
        sumFeaAA = np.reshape(sumFeaAA,(-1,1)) #tek boyutlu diziyi iki boyutlu diziye çevirir
        
        D = sumFeaAA[...,:]+sumFeaAA.transpose()-2*sumFeaAB
        D[D<0]=0
        if(bSqrt==1):
            D = np.sqrt(D)
        D = np.maximum(D,D.transpose())
    else:
        squareFeaA = fea_a*fea_a#Eleman eleman çarpimi
        sumFeaAA = np.sum(squareFeaA,axis=1) ## satir elemanlarini toplar
            
        squareFeaB = fea_b*fea_b#Eleman eleman çarpimi
        sumFeaBB = np.sum(squareFeaB,axis=1)
            
        sumFeaAB = np.dot(fea_a,fea_b.transpose())
        if(isspmatrix(np.array(sumFeaAA))):
            sumFeaAA.toarray()
            sumFeaBB.toarray()
        sumFeaAA = np.reshape(sumFeaAA,(-1,1)) #tek boyutlu diziyi iki boyutlu diziye çevirir
        sumFeaBB = np.reshape(sumFeaBB,(-1,1)) #tek boyutlu diziyi iki boyutlu diziye çevirir
        
        D = sumFeaAA[...,:]+sumFeaBB.transpose()-2*sumFeaAB
        D[D<0]=0
        if(bSqrt==1):
            D = np.sqrt(D)
    
    return D

def NormalizeFea(fea,row):
    
    if row==1:
        nSmp = fea.shape[0]
        feaNorm = np.maximum(1e-14,np.sum(fea**2,1))
        fea = spdiags(feaNorm**-.5,0,nSmp,nSmp)*fea
    else:
        nSmp = fea.shape[1]
        feaNorm = np.maximum(1e-14,np.transpose(np.sum(fea**2,0)))
        fea = fea*spdiags(feaNorm**-.5,0,nSmp,nSmp)
    return fea

def constructWKLPP(fea,options):
    
    bSpeed  = 1
    if 'bNormalized' not in options:
        options['bNormalized'] = 0    
    
    if 'NeighborMode' not in options:
        options['NeighborMode'] = 'KNN'
    
    if options['NeighborMode'] == 'KNN':
        if 'k' not in options:
            options['k'] = 5
        
    elif options['NeighborMode'] == 'Supervised':
        if 'bLDA' not in options:
            options['bLDA'] = 0
        if options['bLDA'] == 1:
            options['bSelfConnected'] = 1
        if 'k' not in options:
            options['k'] = 0
        if 'gnd' not in options:
            print('Label(gnd) should be provided under ''Supervised'' NeighborMode!')
    else:
        print('NeighborMode does not exist!')
        
    if 'WeightMode' not in options:
        options['WeightMode'] = 'HeatKernel'
    
    bBinary = 0
    bCosine = 0
    
    if options['WeightMode'] == 'Binary':
        bBinary = 1
    elif options['WeightMode'] == 'HeatKernel':
        if 't' not in options:
            nSmp = fea.shape[0]
            if nSmp > 3000:
                D = EuDist2(fea[rand.sample(range(nSmp), 3000)],[],1)
            else:
                D = EuDist2(fea,[],1)
            options['t'] = np.mean(D)
    elif options['WeightMode'] == 'Cosine':
        bCosine = 1
    else:
        print('WeightMode does not exist!')
        
    if 'bSelfConnected' not in options:
        options['bSelfConnected'] = 0
    
    if 'gnd' in options:
        nSmp =len(options['gnd'])
    else:
        nSmp = fea.shape[0]
    
    maxM = 62500000
    blockSize = maxM//(nSmp*3)
    
    if options['NeighborMode'] == 'Supervised':
        label = np.unique(options['gnd'])
        nLabel = len(label)
        if options['bLDA'] == 1:
            G = np.zeros((nSmp,nSmp))
            for idx in range(nLabel):
                classIdx = np.where(options['gnd'] == label[idx])[0]
                for i in classIdx:
                    for j in classIdx:
                        G[i,j] = 1/len(classIdx)
            W = coo_matrix(G)
            return W
        
        if options['WeightMode'] == 'Binary':
            if options['k'] > 0:
                G = np.zeros((nSmp*(options['k']+1),3))
                idNow = 0
                for i in range(nLabel):
                    classIdx = np.where(options['gnd']==label[i])[0]
                    D = EuDist2(fea[classIdx],[],0)
                    idx = np.argsort(D, axis=1)#satir ici siralama
                    idx = idx[:,0:options['k']+1]
                    
                    nSmpClass = len(classIdx)*(options['k']+1)
                    G[idNow:nSmpClass+idNow,0] = nm.repmat(classIdx, 1, options['k']+1)
                    G[idNow:nSmpClass+idNow,1] = classIdx[idx.flatten('F')]
                    G[idNow:nSmpClass+idNow,2] = 1
                    idNow = idNow+nSmpClass
                
                G = coo_matrix((G[:,2],(G[:,0],G[:,1])),shape=(nSmp, nSmp))
                G = np.maximum(G.toarray(),G.toarray().transpose())
                G = coo_matrix(G)
            else:
                G = np.zeros((nSmp,nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options['gnd']==label[i])[0]
                    for i in classIdx:
                        for j in classIdx:
                            G[i,j] = 1
            if options['bSelfConnected'] == 0:
                if isinstance(G, (np.ndarray)) :
                    pass
                else:
                    G = G.toarray()
                for i in range(G.shape[0]): 
                    G[i,i] = 0
            W = coo_matrix(G)
            
        elif options['WeightMode'] == 'HeatKernel':
            if options['k'] > 0:
                G = np.zeros((nSmp*(options['k']+1),3))
                idNow = 0
                for i in range(nLabel):
                    classIdx = np.where(options['gnd']==label[i])[0]
                    D = EuDist2(fea[classIdx],[],0)
                    idx = np.argsort(D, axis=1)
                    D.sort(axis=1)
                    idx = idx[:,0:options['k']+1]
                    dump = D[:,0:options['k']+1]
                    dump = np.exp(-dump/(2*np.power(options['t'],2)))
                    
                    nSmpClass = len(classIdx)*(options['k']+1)
                    G[idNow:nSmpClass+idNow,0] = nm.repmat(classIdx, 1, options['k']+1)
                    G[idNow:nSmpClass+idNow,1] = classIdx[idx.flatten('F')]
                    G[idNow:nSmpClass+idNow,2] = dump.flatten('F')
                    idNow = idNow+nSmpClass
                
                G = coo_matrix((G[:,2],(G[:,0],G[:,1])),shape=(nSmp, nSmp))
            else:
                G = np.zeros((nSmp,nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options['gnd']==label[i])[0]
                    D = EuDist2(fea[classIdx],[],0)
                    D = np.exp(-D/(2*np.power(options['t'],2)))
                    for j,m in enumerate(classIdx):
                        for k,n in enumerate(classIdx):
                            G[m,n] = D[j,k] 
                    
            if options['bSelfConnected'] == 0:
                if isinstance(G, (np.ndarray)) :
                    pass
                else:
                    G = G.toarray()
                for i in range(G.shape[0]): 
                    G[i,i] = 0
            
            W = coo_matrix(np.maximum(G,G.transpose()))
        elif options['WeightMode'] == 'Cosine':
            if options['bNormalized'] == 0: 
                fea = NormalizeFea(fea,1)
            if options['k'] > 0:
                G = np.zeros((nSmp*(options['k']+1),3))
                idNow = 0
                for i in range(nLabel):
                    classIdx = np.where(options['gnd']==label[i])[0]
                    D = np.dot(fea[classIdx],np.transpose(fea[classIdx]))
                    idx = np.argsort(-D, axis=1)#satir ici siralama
                    D = -D
                    D.sort(axis=1)
                    idx = idx[:,0:options['k']+1]
                    dump = -D[:,0:options['k']+1]
                    
                    nSmpClass = len(classIdx)*(options['k']+1)
                    G[idNow:nSmpClass+idNow,0] = nm.repmat(classIdx, 1, options['k']+1)
                    G[idNow:nSmpClass+idNow,1] = classIdx[idx.flatten('F')]
                    G[idNow:nSmpClass+idNow,2] = dump.flatten('F')
                    idNow = idNow+nSmpClass
                G = coo_matrix((G[:,2],(G[:,0],G[:,1])),shape=(nSmp, nSmp))
            else:
                G = np.zeros((nSmp,nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options['gnd']==label[i])[0]
                    D = np.dot(fea[classIdx,:],np.transpose(fea[classIdx,:]))
                    for j,m in enumerate(classIdx):
                        for k,n in enumerate(classIdx):
                            G[m,n] = D[j,k]
            if options['bSelfConnected'] == 0:
                if isinstance(G, (np.ndarray)) :
                    pass
                else:
                    G = G.toarray()
                for i in range(G.shape[0]): 
                    G[i,i] = 0
            W = coo_matrix(np.maximum(G,G.transpose()))
        else:
            print('WeightMode does not exist!')
        return W
    if (bCosine == 1 and options['bNormalized']==0):
        Normfea = NormalizeFea(fea,1)
    if options['NeighborMode'] == 'KNN' and options['k']>0:
        if not (bCosine and options['bNormalized']):
            G = np.zeros((nSmp*(options['k']+1),3))
            for i in range(math.ceil(nSmp/blockSize)):
                if i == math.ceil(nSmp/blockSize) - 1:
                    smpIdx = np.arange(i*blockSize,nSmp)
                    dist = EuDist2(fea[smpIdx,:],fea,0)
                    if bSpeed == 1:
                        nSmpNow = len(smpIdx);
                        dump = np.zeros((nSmpNow,options['k']+1))
                        idx = np.zeros((nSmpNow,options['k']+1))
                        for j in range(options['k']+1):
                            dump[:,j] = np.min(dist, axis=1)
                            idx[:,j] = np.argmin(dist, axis=1)
                            temp = idx[:,j]*nSmpNow+np.arange(nSmpNow)
                            sizeTemp = dist.shape[0]
                            sizeTemp2 = dist.shape[1]
                            tempDist = dist.flatten('F')
                            for x in temp:
                                tempDist[int(x)] = np.exp(100)
                            dist = tempDist.reshape(sizeTemp2,sizeTemp)
                            dist = np.transpose(dist)
                    else:
                        idx = np.argsort(dist, axis=1)#satir ici siralama
                        dist.sort(axis=1)
                        dump = dist[:,0:options['k']+1]
                        idx = idx[:,0:options['k']+1]
                    if bBinary == 0:
                        if bCosine == 1:
                            dist = np.dot(Normfea[smpIdx,:],np.transpose(Normfea))
                            linidx = np.transpose(np.arange(0,len(idx))).reshape(-1,1)
                            sub2ind = idx*len(dist)+linidx[:,[0]*len(idx[0])]
                            tempDistance = dist.flatten('F')
                            for rr in range(len(sub2ind[0])):
                                dump[:,rr] = tempDistance[np.int_(sub2ind[:,rr])]
                        else:
                            dump = np.exp(-dump/(2*np.power(options['t'],2)))
                        G[np.arange(i*blockSize*(options['k']+1),nSmp*(options['k']+1)),0] = nm.repmat(smpIdx.reshape(-1,1),options['k']+1,1).flatten('F')
                        G[np.arange(i*blockSize*(options['k']+1),nSmp*(options['k']+1)),1] = idx.flatten('F')
                        if bBinary == 0:
                            G[np.arange(i*blockSize*(options['k']+1),nSmp*(options['k']+1)),2] = dump.flatten('F')
                        else:
                            G[np.arange(i*blockSize*(options['k']+1),nSmp*(options['k']+1)),2] = 1
                else:
                    smpIdx = np.arange(i*blockSize,(i+1)*blockSize) 
                    dist = EuDist2(fea[smpIdx,:],fea,0)
                    
                    if bSpeed == 1:
                        nSmpNow = len(smpIdx);
                        dump = np.zeros((nSmpNow,options['k']+1))
                        idx = np.zeros((nSmpNow,options['k']+1))
                        for j in range(options['k']+1):
                            dump[:,j] = np.min(dist, axis=1)
                            idx[:,j] = np.argmin(dist, axis=1)
                            temp = idx[:,j]*nSmpNow+np.arange(nSmpNow)
                            sizeTemp = dist.shape[0]
                            sizeTemp2 = dist.shape[1]
                            tempDist = dist.flatten('F')
                            for x in temp:
                                tempDist[int(x)] = np.exp(100)
                            dist = tempDist.reshape(sizeTemp2,sizeTemp)
                            dist = np.transpose(dist)
                    else:
                        idx = np.argsort(dist, axis=1)#satir ici siralama
                        dist.sort(axis=1)
                        dump = dist[:,0:options['k']+1]
                        idx = idx[:,0:options['k']+1]
                    if bBinary == 0:
                        if bCosine == 1:
                            dist = np.dot(Normfea[smpIdx,:],np.transpose(Normfea))
                            linidx = np.transpose(np.arange(0,len(idx))).reshape(-1,1)
                            sub2ind = idx*len(dist)+linidx[:,[0]*len(idx[0])]
                            tempDistance = dist.flatten('F')
                            for rr in range(len(sub2ind[0])):
                                dump[:,rr] = tempDistance[np.int_(sub2ind[:,rr])]
                        else:
                            dump = np.exp(-dump/(2*np.power(options['t'],2)))
                        G[np.arange(i*blockSize*(options['k']+1),(i+1)*blockSize*(options['k']+1)),0] = nm.repmat(smpIdx.reshape(-1,1),options['k']+1,1).flatten('F')
                        G[np.arange(i*blockSize*(options['k']+1),(i+1)*blockSize*(options['k']+1)),1] = idx.flatten('F')
                        if bBinary == 0:
                            G[np.arange(i*blockSize*(options['k']+1),(i+1)*blockSize*(options['k']+1)),2] = dump.flatten('F')
                        else:
                            G[np.arange(i*blockSize*(options['k']+1),(i+1)*blockSize*(options['k']+1)),2] = 1
            W = coo_matrix((G[:,2],(G[:,0],G[:,1])),shape=(nSmp, nSmp))
        else:
            G = np.zeros((nSmp*(options['k']+1),3))
            for i in range(math.ceil(nSmp/blockSize)):
                if i == math.ceil(nSmp/blockSize) - 1:
                    smpIdx = np.arange(i*blockSize,nSmp)
                    dist = np.dot(fea[smpIdx,:],np.transpose(fea))
                    if bSpeed == 1:
                        nSmpNow = len(smpIdx);
                        dump = np.zeros((nSmpNow,options['k']+1))
                        idx = np.zeros((nSmpNow,options['k']+1))
                        for j in range(options['k']+1):
                            dump[:,j] = np.max(dist, axis=1)
                            idx[:,j] = np.argmax(dist, axis=1)
                            temp = idx[:,j]*nSmpNow+np.arange(nSmpNow)
                            sizeTemp = dist.shape[0]
                            sizeTemp2 = dist.shape[1]
                            tempDist = dist.flatten('F')
                            for x in temp:
                                tempDist[int(x)] = 0
                            dist = tempDist.reshape(sizeTemp2,sizeTemp)
                            dist = np.transpose(dist)
                    else:
                        idx = np.argsort(-dist, axis=1)#satir ici siralama
                        dist = -dist
                        dist.sort(axis=1)
                        idx = idx[:,0:options['k']+1]
                        dump = -dist[:,0:options['k']+1]
                    G[np.arange(i*blockSize*(options['k']+1),nSmp*(options['k']+1)),0] = nm.repmat(smpIdx.reshape(-1,1),options['k']+1,1).flatten('F')
                    G[np.arange(i*blockSize*(options['k']+1),nSmp*(options['k']+1)),1] = idx.flatten('F')
                    G[np.arange(i*blockSize*(options['k']+1),nSmp*(options['k']+1)),2] = dump.flatten('F')
                else:
                    smpIdx = np.arange(i*blockSize,(i+1)*blockSize)
                    dist = np.dot(fea[smpIdx,:],np.transpose(fea))
                    if bSpeed == 1:
                        nSmpNow = len(smpIdx);
                        dump = np.zeros((nSmpNow,options['k']+1))
                        idx = np.zeros((nSmpNow,options['k']+1))
                        for j in range(options['k']+1):
                            dump[:,j] = np.max(dist, axis=1)
                            idx[:,j] = np.argmax(dist, axis=1)
                            temp = idx[:,j]*nSmpNow+np.arange(nSmpNow)
                            sizeTemp = dist.shape[0]
                            sizeTemp2 = dist.shape[1]
                            tempDist = dist.flatten('F')
                            for x in temp:
                                tempDist[int(x)] = 0
                            dist = tempDist.reshape(sizeTemp2,sizeTemp)
                            dist = np.transpose(dist)
                    else:
                        idx = np.argsort(-dist, axis=1)#satir ici siralama
                        dist = -dist
                        dist.sort(axis=1)
                        idx = idx[:,0:options['k']+1]
                        dump = -dist[:,0:options['k']+1]
                    G[np.arange(i*blockSize*(options['k']+1),(i+1)*blockSize*(options['k']+1)),0] = nm.repmat(smpIdx.reshape(-1,1),options['k']+1,1).flatten('F')
                    G[np.arange(i*blockSize*(options['k']+1),(i+1)*blockSize*(options['k']+1)),1] = idx.flatten('F')
                    G[np.arange(i*blockSize*(options['k']+1),(i+1)*blockSize*(options['k']+1)),2] = dump.flatten('F')
            W = coo_matrix((G[:,2],(G[:,0],G[:,1])),shape=(nSmp, nSmp))
        if bBinary == 1:
            W[W!=0]=1
        if options['bSelfConnected'] == 0:
            W = W - np.diag(np.diag(W.toarray()))
        if 'bTrueKNN' in options:
            pass
        else:
            W = np.maximum(W,np.transpose(W))
        return W
        

def constructKernelKLPP(fea_a,fea_b,options):
#           KernelType  -  Choices are:
#               'Gaussian'      - e^{-(|x-y|^2)/2t^2}
#               'Polynomial'    - (x'*y)^d
#               'PolyPlus'      - (x'*y+1)^d
#               'Linear'        -  x'*y
#
#               t       -  parameter for Gaussian
#               d       -  parameter for Poly
    
    if 'KernelType' in options:
        pass
    else:
        options['KernelType']='Gaussian'
        
    ##=======================================##
    if(options['KernelType']=='Gaussian'):
        if 't' in options:
            pass
        else:
            options['t']=1
    elif(options['KernelType']=='Polynomial'):
        if 'd' in options:
            pass
        else:
            options['d']=2
    elif(options['KernelType']=='PolyPlus'):
        if 'd' in options:
            pass
        else:
            options['d']=2
    elif(options['KernelType']=='Linear'):
        pass
    else:
        print('KernelType does not exist!')
        
    ##=======================================##
    if(options['KernelType']=='Gaussian'):
        if(len(fea_b)==0):
            D = EuDist2(fea_a, [], 0)
        else:
            D = EuDist2(fea_a, fea_b, 0)
        K = np.exp(-D/(2*np.power(options['t'],2)))
    elif(options['KernelType']=='Polynomial'):
        if(len(fea_b)==0):
            D = np.dot(fea_a,fea_a.transpose())
        else:
            D = np.dot(fea_a,fea_b.transpose())
        K = np.power(D,options['d'])
    elif(options['KernelType']=='PolyPlus'):
        if(len(fea_b)==0):
            D = np.dot(fea_a,fea_a.transpose())
        else:
            D = np.dot(fea_a,fea_b.transpose())
        K = np.power(np.add(D,1),options['d'])
    elif(options['KernelType']=='Linear'):
        if(len(fea_b)==0):
            K = np.dot(fea_a,fea_a.transpose())
        else:
            K = np.dot(fea_a,fea_b.transpose())
    else:
        print('KernelType does not exist!')
    
    if len(fea_b)==0:
        K = np.maximum(K,np.array(K).transpose())
        
    return K
           

# index = 0
# fea = np.ones((50,10))
# with open("C:\\Users\\osivaz\\Desktop\\TouchScreen\\Touch_BenchmarkSerwaddaReadAndDRs\\classification\\klpp\\fea.txt", "r") as file:
#     for line in file.readlines():
#         f_list = [float(i) for i in line.split(",")]
#         fea[index,:] = f_list
#         index += 1    

fea = np.random.rand(50,40)
gnd = [1]*10 + [2]*15 + [3]*10 + [4]*15
options={}
options['NeighborMode'] = 'Supervised'
options['gnd'] = gnd
options['WeightMode'] = 'HeatKernel'
#options['bNormalized'] = 1
#options['bLDA'] = 1
options['t'] = 25
options['k'] = 5
options['reducedDim'] = 2
W = constructWKLPP(fea,options)
options['KernelType'] = 'Gaussian'
options['Regu'] = 1
options['ReguAlpha'] = 0.01
eigvector, eigvalue = KLPP(W, options, fea)
kTrain = constructKernelKLPP(fea,[],options)
train = np.dot(kTrain,eigvector)