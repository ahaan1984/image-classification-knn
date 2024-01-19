import math
import statistics
import numpy as np

class KNN:
    def __init__(self, metric, k=3, p=None):
        self.k = k
        self.metric = metric 
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        preds = []
        for test_row in X_test:
            nn = self.get_neighbours(test_row)
            majority = statistics.mode([n[1] if not np.isscalar(n) else n for n in nn])
            preds.append(majority)
        return np.array(preds)

    def euclidean(self, dp1, dp2):
        return np.sqrt(np.sum((dp1 - dp2)**2))
    
    def manhattan(self, dp1, dp2):
        return np.sum(np.abs(dp1-dp2))
    
    def minkowski(self, dp1, dp2, p=2):
        return np.sum(np.abs(dp1-dp2)**p)**(1/p)
    
    def get_neighbours(self, test_row):
        distances = []
        for (train_row, train_class) in zip(self.X_train, self.y_train):
            if self.metric == 'euclidean':
                dist = self.euclidean(train_row, test_row)
            elif self.metric == 'manhattan':
                dist = self.manhattan(train_row, test_row)
            elif self.metric == 'minkowski':
                dist = self.minkowski(train_row, test_row)
            else:
                raise NameError('Supported metrics are euclidean, manhattan and minkowski.')
            distances.append((dist, train_class))
            neighbours = list()
            for i in range(min(self.k, len(distances))):
                neighbours.append(distances[i][1])

            return neighbours