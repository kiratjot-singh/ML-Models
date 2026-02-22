import numpy as np 
class KMeans:
    def __init__(self,n=2,max_iter=100):
        self.n=n
        self.max_iter=max_iter
        self.centroids=None
    def fit(self,X):
        

