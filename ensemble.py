import pickle
import math
import numpy as np
from sklearn import tree


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.Weaker = weak_classifier
        self.W = np.ones((800,)) * 1/800
        self.M = n_weakers_limit
        self.Q = 0

    def is_good_enough(self):
        '''Optional'''
        self.sums = self.sums + self.G[self.Q].predict(self.X)*self.alpha[self.Q]
        for i in range(self.y.shape[0]):
            if self.sums[i] >0.05:
                self.sums[i] = 1
            else:
                self.sums[i] = 0
        t = (self.sums != self.y).sum()
        return t==0

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.X = X
        self.y = y
        self.sums=np.zeros(self.y.shape)
        self.G = {}
        self.alpha = {}
        for i in range(self.M):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
    
        for i in range(self.M):
            self.G[i] = tree.DecisionTreeClassifier(max_depth=3 )
            self.G[i] = self.G[i].fit(X,y,self.W)
           
           
            e = 0
            predict = self.G[i].predict(X)
            for j in range(800):
                e = e + self.W[j] * (not(predict[j] == y[j]))
    
            self.alpha[i] = math.log((1-e)/e)
            Z =np.dot(self.W , np.exp(-self.alpha[i]*predict))
            self.W = self.W/Z * np.exp(-self.alpha[i]*predict).T
            
            self.Q = i
            if self.is_good_enough() == 1:
                print(i+1," weak classifier is enough to  make the error to 0")    
                break
            
        s = sum(self.alpha)
        for i in range(self.Q):
            self.alpha[i] = self.alpha[i]/s
            
        return self.G


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        sums = np.zeros(X.shape[0])
        for i in range(self.Q):
            sums = sums + self.G[i].predict(X)*self.alpha[i]
        return sums

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        predicts = self.predict_scores(X)
        for i in range(X.shape[0]):
            if predicts[i] > threshold:
                predicts[i] = 1
            else:
                predicts[i] = 0
        return predicts

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
