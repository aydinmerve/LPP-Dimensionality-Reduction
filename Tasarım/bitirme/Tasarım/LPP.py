# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:36:17 2019


"""
# kütüphaneleri import ediyoruz
import numpy as np
import random as rand
import numpy.matlib as nm
import math
from scipy.sparse import coo_matrix, isspmatrix, spdiags
from numpy.linalg import cholesky, LinAlgError, norm
from scipy.sparse.linalg import eigs, eigsh

def CutonRatio(U,S,V,options):
    
    if 'PCARatio' in options:
        pass
    else:
        options['PCARatio'] = 1
    eigvalue_PCA = np.diag(S.toarray())
    if options['PCARatio'] > 1:
        idx = options['PCARatio']
        if idx < len(eigvalue_PCA):
            U = U[:,0:idx]
            V = V[:,0:idx]
            S = np.diag(S.toarray()[0:idx,0:idx])
            S = spdiags(S,0,len(S),len(S))
    elif options['PCARatio'] < 1:
        sumEig = sum(eigvalue_PCA)
        sumEig = sumEig*options['PCARatio']
        sumNow = 0
        for idx in range(len(eigvalue_PCA)):
            sumNow = sumNow + eigvalue_PCA[idx]
            if sumNow >= sumEig:
                break
        U = U[:,0:idx]
        V = V[:,0:idx]
        S = np.diag(S.toarray()[0:idx,0:idx])
        S = spdiags(S,0,len(S),len(S))
    return U, S, V


def mySVD(X,nargout):
    
    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1
    
    reducedDim = 0

    nSmp, mFea = np.shape(X)
    
    if (mFea/nSmp > 1.0713):
        # This is an efficient method which computes the eigvectors of
	    # of A*A^T (instead of A^T*A) first, and then convert them back to
	    # the eigenvectors of A^T*A.
        ddata = np.dot(X,X.transpose())
        ddata = np.maximum(ddata,ddata.transpose())
    
        dimMatrix = len(ddata)
        if(reducedDim > 0 & dimMatrix > MAX_MATRIX_SIZE & reducedDim < dimMatrix*EIGVECTOR_RATIO):
            eigvalue, U = np.linalg.eig(ddata)
            index = np.argsort(-eigvalue)
            eigvalue = eigvalue[index]
            U = U[:,index]
            eigvalue = np.array(eigvalue[0:reducedDim])
            U = U[:,0:reducedDim]
        else:
            if isspmatrix(ddata):
                ddata = ddata.toarray()
            eigvalue, U = np.linalg.eig(ddata)
            index = np.argsort(-eigvalue)
            eigvalue = eigvalue[index]
            U = U[:,index]
        maxEigValue = max(np.abs(eigvalue))
        eigIdx = np.where(abs(eigvalue)/maxEigValue > 1e-10)[0]## 'where' return tuple,with [0] gettting array
        eigvalue = eigvalue[eigIdx]
        U = U[:,eigIdx]
        
        if reducedDim > 0 & reducedDim < len(eigvalue):
            eigvalue = eigvalue[0:reducedDim]
            U = U[:,0:reducedDim]
            
        eigvalue_Half = np.sqrt(eigvalue)
        S =  spdiags(eigvalue_Half,0,len(eigvalue_Half),len(eigvalue_Half))
        if nargout >= 3:
            eigvalue_MinusHalf = np.power(eigvalue_Half,-1)
            V = np.dot(np.array(X).transpose(),(np.array(U)*nm.repmat(eigvalue_MinusHalf,len(U),1)))
    else:
        ddata = np.dot(X.transpose(),X)
        ddata = np.maximum(ddata,ddata.transpose())
    
        dimMatrix = len(ddata)
        if(reducedDim > 0 & dimMatrix > MAX_MATRIX_SIZE & reducedDim < dimMatrix*EIGVECTOR_RATIO):
            eigvalue, V = np.linalg.eig(ddata)
            index = np.argsort(-eigvalue)
            eigvalue = eigvalue[index]
            V = V[:,index]
            eigvalue = np.array(eigvalue[0:reducedDim])
            V = V[:,0:reducedDim]
        else:
            if isspmatrix(ddata):
                ddata = ddata.toarray()
            eigvalue, V = np.linalg.eig(ddata)
            index = np.argsort(-eigvalue)
            eigvalue = eigvalue[index]
            V = V[:,index]
        maxEigValue = max(np.abs(eigvalue))
        eigIdx = np.where(abs(eigvalue)/maxEigValue > 1e-10)[0]## 'where' return tuple,with [0] gettting array
        eigvalue = eigvalue[eigIdx]
        V = V[:,eigIdx]
        
        if reducedDim > 0 & reducedDim < len(eigvalue):
            eigvalue = eigvalue[0:reducedDim]
            V = V[:,0:reducedDim]
            
        eigvalue_Half = np.sqrt(eigvalue)
        S =  spdiags(eigvalue_Half,0,len(eigvalue_Half),len(eigvalue_Half))
        
        eigvalue_MinusHalf = np.power(eigvalue_Half,-1)
        U = np.dot(np.array(X),(np.array(V)*nm.repmat(eigvalue_MinusHalf,len(V),1)))
           
    return U,S,V


def LGE(W, D, options, data):
    
    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1
    
    reducedDim = 30
    if 'reducedDim' in options:
        reducedDim = options['reducedDim']
    if 'Regu' not in options:
        bPCA = 1
        if 'PCARatio' not in options:
            options['PCARatio'] = 1
    else:
        bPCA = 0
        if 'ReguType' not in options:
            options['ReguType'] = 'Ridge'
        if 'ReguAlpha' not in options:
            options['ReguAlpha'] = 0.1
        if 'PCARatio' not in options:
            options['PCARatio'] = 1
    bD = 1
    
    if(isinstance(D, (np.ndarray))):
        lenD = len(D)
    else:
        
        lenD = len(D.toarray())
    
    if lenD == 0:
        bD = 0
    nSmp,nFea = np.shape(data)
    if W.shape[0] != nSmp:
        print('W and data mismatch!')
    if bD ==1 & (lenD != nSmp):
        print('D and data mismatch!')
    
    bChol = 0
    if bPCA==1 & (nSmp > nFea) & (options['PCARatio'] >= 1):
        if bD == 1:
            DPrime = np.dot(np.dot(data.transpose(),D),data)
        else:
            DPrime = np.dot(data.transpose(),data)
        DPrime = np.maximum(DPrime, DPrime.transpose())
        try:
            R = cholesky(DPrime)
            p = True
        except LinAlgError as err:
            if 'Matrix is not positive definite' in str(err):
                p = False
        if p == True:
            bPCA = 0
            bChol = 1
    if bPCA == 1:
        U, S, V = mySVD(data,3)
        U, S ,V = CutonRatio(U,S,V,options)
        eigvalue_PCA = np.diag(S.toarray())
        if bD == 1:
            data = np.dot(U,S)
            eigvector_PCA = V
            
            DPrime = np.dot(np.dot(data.transpose(),D),data)
            DPrime = np.maximum(DPrime, DPrime.transpose())
        else:
            data = U
            K = spdiags(np.power(eigvalue_PCA,-1), 0, len(eigvalue_PCA), len(eigvalue_PCA))
            eigvector_PCA = np.dot(V,K.toarray())
    else:
        if bChol == 0:
            if bD == 1:
                DPrime = np.dot(np.dot(data.transpose(),D.toarray()),data)
            else:
                DPrime = np.dot(data.transpose(),data)
            if options['ReguType'] == 'Ridge':
                for i in range(len(DPrime)):
                    DPrime[i,i] += options['ReguAlpha']
            elif options['ReguType'] == 'Tensor':
                DPrime = np.add(DPrime, options['ReguAlpha']*options['regularizerR'])
            elif options['ReguType'] == 'Custom':
                DPrime = np.add(DPrime, options['ReguAlpha']*options['regularizerR'])
            else:
                print('ReguType does not exist!')
                
            DPrime = np.maximum(DPrime,DPrime.transpose())
    
    WPrime = np.dot(np.dot(data.transpose(),W.toarray()),data)
    WPrime = np.maximum(WPrime,WPrime.transpose())

    dimMatrix = WPrime.shape[0]
    if reducedDim > dimMatrix:
        reducedDim = dimMatrix   
    
    if 'bEigs' in options:
        bEigs = options['bEigs']
    else:
        if (dimMatrix > MAX_MATRIX_SIZE) & (reducedDim < dimMatrix*EIGVECTOR_RATIO):
            bEigs = 1
        else:
            bEigs = 0
    
    if bEigs == 1:
        if bPCA ==1 and bD == 0:
            eigvalue, eigvector = eigsh(WPrime, k=reducedDim, which='LA')
        else:
            if bChol ==1:
                eigvalue, eigvector = eigsh(WPrime, k=reducedDim, M=R.transpose(), which='LA')
            else:
                eigvalue, eigvector = eigsh(WPrime, k=reducedDim, M=DPrime, which='LA')
    else:
        if bPCA==1 and bD == 0:
            eigvalue, eigvector = eigsh(WPrime, k = len(WPrime)-1)
        else:
            eigvalue, eigvector = eigsh(DPrime - WPrime, k = len(WPrime)-1, M=DPrime)
        index = np.argsort(eigvalue)
        eigvalue = eigvalue[index]
        eigvector = eigvector[:,index]
        
        if reducedDim < eigvector.shape[1]:
            eigvector = eigvector[:,0:reducedDim]
            eigvalue = eigvalue[0:reducedDim]
    
    if bPCA == 1:
        eigvector = np.dot(eigvector_PCA,eigvector)
    
    for i in range(eigvector.shape[1]):
        eigvector[:,i] = eigvector[:,i]/norm(eigvector[:,i])
    
    return eigvector, eigvalue
            

def LPP(W, options, data):
    
    nSmp,nFea = np.shape(data)
    if W.shape[0] != nSmp:
        print('W and data mismatch!')
    
    if 'keepMean' in options:
        pass
    else:
        if isspmatrix(data):
            data = data.toarray()
        sampleMean = np.mean(data, axis = 0)
        data = (data - nm.repmat(sampleMean,nSmp,1))
        
    D = np.sum(W,axis=1)
    
    if 'Regu' not in options:
        DToPowerHalf = np.power(D,0.5)
        D_mhalf = np.power(DToPowerHalf,-1)
        #D_mhalf = np.power(np.add(DToPowerHalf,0.01),-1)
        
        if nSmp < 5000:
            tmpD_mhalf = nm.repmat(D_mhalf,1,nSmp)
            if isinstance(W, (np.ndarray)):
                W = np.multiply(np.multiply(tmpD_mhalf,W),tmpD_mhalf.transpose())
            else:
                W = np.multiply(np.multiply(tmpD_mhalf,W.toarray()),tmpD_mhalf.transpose())
            W = coo_matrix(W)
        else:
            j_idx, i_idx = np.where(W.toarray() != 0)
            v1_idx = np.zeros(len(i_idx))
            LL = W.toarray()
            for i in range(len(j_idx)):
                v1_idx[i] = LL[i_idx[i],j_idx[i]]*D_mhalf[i_idx[i]]*D_mhalf[j_idx[i]]
            W = coo_matrix((v1_idx,(i_idx,j_idx)), shape=(nSmp, nSmp))
        
        W = np.maximum(W.toarray(),W.toarray().transpose())
        W = coo_matrix(W)
        data = np.multiply(nm.repmat(DToPowerHalf,1,nFea),data)
        eigvector, eigvalue = LGE(W, np.array([]), options, data)
    else:
        if 'ReguAlpha' not in options:
            options['ReguAlpha'] = 0.1
        options['ReguAlpha'] = options['ReguAlpha']*sum(D)/len(D)
        D = coo_matrix((np.array(D).ravel(),(range(nSmp),range(nSmp))),shape = (nSmp,nSmp))
        eigvector, eigvalue = LGE(W, D, options, data)
    
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

def constructWLPP(fea,options):
    
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
                    

