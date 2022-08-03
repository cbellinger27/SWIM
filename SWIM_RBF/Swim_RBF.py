import numpy as np
from scipy.spatial.distance import pdist
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler


def rbf(d, eps):
    return np.exp(-(d * eps) ** 2)


def distance(x, y):
    return np.sum(np.abs(x - y))


def pairwise_distances(X):
    D = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue

            d = distance(X[i], X[j])

            D[i][j] = d
            D[j][i] = d

    return D


def score(point, X, epsilon):
    mutual_density_score = 0.0

    for i in range(len(X)):
        rbfRes = rbf(distance(point, X[i,:]), epsilon)
        mutual_density_score += rbfRes

    return mutual_density_score

def scoreAll(points, X, epsilon):
    cur_mutual_density_score = 0.0
    mutual_density_scores    = []

    for j in range(len(points)):
        point = points[j, :]

        for i in range(len(X)):
            rbfRes = rbf(distance(point, X[i,:]), epsilon)
            cur_mutual_density_score += rbfRes

        mutual_density_scores = np.append(mutual_density_scores, cur_mutual_density_score)
        cur_mutual_density_score = 0.0

    return mutual_density_scores



class SwimRBF:
    def __init__(self, minCls=None, epsilon=None, steps=5, tau = 0.25):
        self.epsilon = epsilon
        self.steps   = steps
        self.tau     = tau
        self.minCls  = minCls
        self.scaler = StandardScaler()

    def extremeRBOSample(self, data, labels, numSamples):

        if self.minCls == None:
            self.minCls = np.argmin(np.bincount(labels.astype(int)))
        
        trnMajData = data[np.where(labels!=self.minCls)[0], :]
        trnMinData = data[np.where(labels==self.minCls)[0], :]
        
        if self.epsilon == None:
            self.epsilon = self.fit(trnMajData)

        synthData  = np.empty([0, data.shape[1]])
        stds       = self.tau*np.std(trnMinData, axis=0)

        if(np.sum(labels==self.minCls)==1):
            trnMinData    = trnMinData.reshape(1,len(trnMinData))

        while synthData.shape[0] < numSamples:
            j     = np.random.choice(trnMinData.shape[0],1)[0]
            scoreCur = score(trnMinData[j,:], trnMajData,  self.epsilon)
            for k in range(self.steps):
                step      = trnMinData[j,:] + np.random.normal(0, stds, trnMinData.shape[1])
                stepScore = score(step, trnMajData,  self.epsilon)
                if stepScore <= scoreCur:
                    synthData = np.append(synthData,step.T.reshape((1, len(step))),axis=0)
                    break
        
        sampled_data = np.concatenate([np.array(synthData), data])
        sampled_labels = np.append([self.minCls]*len(synthData),labels)

        return sampled_data, sampled_labels
    
    def fit(self, data):
        d = pdist(data) 
        return 0.5 * np.std(d) * np.mean(d)


