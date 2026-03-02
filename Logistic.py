import numpy as np

class Logistic:
    def __init__(self,lr,epochs):
        self.lr=lr
        self.epochs=epochs
        self.bitta=None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

    def fit(self,X,y):
        X=np.insert(X,0,1,axis=1)
        self.bitta=np.ones(X.shape[1])
        for i in range(self.epochs):
            y_hat=self.sigmoid(np.dot(X,self.bitta))
            gradient=(1/X.shape[0])*np.dot(X.T,y_hat-y)
            self.bitta=self.bitta-self.lr*gradient
    def predict(self,X_test):
        X_test = np.insert(X_test, 0, 1, axis=1)
        probs = self.sigmoid(np.dot(X_test, self.bitta))
        return (probs >= 0.5).astype(int)