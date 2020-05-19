# Copyright 2018, Colin Bellinger, All rights reserved.
# paper "Synthetic oversampling with the majority class: A new perspective on handling extreme imbalance".
# IEEE 2018 International Converence on Data Mining 
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

class SingularMatrixException(Exception):
    def __init__(self):
        Exception.__init__(self,"Singular data matrix... use subspace") 

def _msqrt(X):
    '''Computes the square root matrix of symmetric square matrix X.'''
    (L, V) = np.linalg.eig(X)
    return V.dot(np.diag(np.sqrt(L))).dot(V.T) 


class SwimMaha:

    def __init__(self, sd=0.25, minClass=None, subSpaceSampling=False):
        self.sd = sd
        self.minClass = minClass
        self.subSpaceSampling = subSpaceSampling

    # the data passed is transposed, so the rows are the features, and the columns are the instances
    def mahaSampling(self, data, labels, numSamples):

        if self.minClass == None:
            self.minClass     = np.argmin(np.bincount(labels.astype(int)))

        syntheticInstances  = []
        data_maj_orig       = data[np.where(labels!=self.minClass)[0], :]
        data_min_orig       = data[np.where(labels==self.minClass)[0], :]

        if(np.sum(labels==self.minClass)==1):
            data_min_orig = data_min_orig.reshape(1,len(data_min_orig))
            # trnMinData    = trnMinData.reshape(1,len(trnMinData))

        ## STEP 1: CENTRE
        ## CENTRE THE MAJORITY CLASS AND CENTRE THE MINORITY CLASS WITH RESPECT TO THE MAJORITY CLASS
        scaler = StandardScaler(with_std=False)
        T_maj  = np.transpose(scaler.fit_transform(data_maj_orig))
        T_min  = np.transpose(data_min_orig) 

        ## STEP 2: WHITEN
        C_inv = None
        C     = np.cov(T_maj) # the covariance matrix - of the majority class

        # CALCULATE THE RANK OF THE MAJORITY CLASS DATA MATRIX AND INVERT IT IF POSSIBLE
        data_rank = np.linalg.matrix_rank(data_maj_orig) 
        if data_rank < T_maj.shape[0]: # there are linearly dependent column, so inverse will be singular
            if self.subSpaceSampling == False:
                print("The majority class has linearly dependent columns. Rerun the sampling subSpaceSampling=True. Return original data.")
                return data, labels
            else:

                QR = np.linalg.qr(data_maj_orig)
                indep = QR[1].diagonal() > 0
                data = data[:,indep]
                print("The majority class has linearly dependent columns. Resampled data will be in the " + str(sum(indep==True)) + " independent columns of the orginal " + str(data_maj_orig.shape[1]) + "-dimensional data.")

        else:
            try:
                C_inv = np.linalg.inv(C) # inverse of the covariance matrix
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    print("Majority class data is singular. Degrading to random oversampling with Gaussian jitter")
                    X_new = data_min_orig[np.random.choice(data_min_orig.shape[0], numSamples, replace=True), :]
                    X_new = X_new + (0.1 * np.random.normal(0, data_maj_orig.std(0), X_new.shape))
                    y_new = np.repeat(self.minClass, numSamples)
                    data   = np.concatenate([X_new, data])
                    labels = np.append(y_new,labels)
                    return data, labels
        
        try:
            M     = _msqrt(C_inv) # C_inv is the inverse of the covariance matrix, and M is the matrix for the whitening transform
            M_inv = np.linalg.inv(M) # this is the inverse of the M matrix, we'll use it for getting the data back.

            W_min      = M.dot(T_min) # whitening transform - whiten the minority class
            W_maj      = M.dot(T_maj) # whitening transform - whiten the majority class
        except:
            print("value excpetion... synthetic instances not generated")
            return data, labels

        ## STEP 3: FIND THE MEANS AND FEATURE BOUNDS TO USE IN THE GENERATION PROCESS
        min_means  = W_min.mean(1)
        min_stds   = W_min.std(1)
        min_ranges_bottom = min_means - self.sd*min_stds
        min_ranges_top    = min_means + self.sd*min_stds

 
        ## STEP 4: GENERATE SYNTHETIC INSTANCES
        # RANDOMLY REPLICATE THE WHITENED MINORITY CLASS INSTNACES <numSamples> TIMES TO GENERATE SYNTHETIC INSTANCES FROM
        smpInitPts = W_min[:, np.random.choice(W_min.shape[1], numSamples)]
        for smpInd in range(smpInitPts.shape[1]): # repeat "times" times, so we get a balanced dataset
            new_w_raw = []
            new       = None
            new_w     = None
            smp       = smpInitPts[:, smpInd]
            for dim in range(len(min_means)):
                new_w_raw.append(random.uniform(smp[dim]-self.sd*min_stds[dim], smp[dim]+self.sd*min_stds[dim]))

            ## Step 5: SCALE BACK TO THE ORIGINAL SPACE
            new_w = np.array(new_w_raw) / ((np.linalg.norm(new_w_raw)/np.linalg.norm(smp)))
            new   = M_inv.dot(np.array(new_w))
               
            syntheticInstances.append(new)
            
        sampled_data   = np.concatenate([np.array(syntheticInstances), data])
        sampled_labels = np.append([self.minClass]*len(syntheticInstances),labels)

        return sampled_data, sampled_labels

