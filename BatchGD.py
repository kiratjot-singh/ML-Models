import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data=load_diabetes()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
class BatchGD():
    def __init__(self,lr=0.01,epochs=1000):
        self.m=None
        self.b=None
        self.lr=lr
        self.epochs=epochs
    def fit(self,X_train,y_train):
        self.m=np.ones(X_train.shape[1])
        self.b=1
        for i in range(self.epochs):
            y_hat=np.dot(X_train,self.m)+self.b
            slope1=-2* np.dot(X_train.T,y_train-y_hat)/X_train.shape[0]
            slope2=-2*np.mean(y_train-y_hat)
            self.m=self.m-self.lr*slope1
            self.b=self.b-self.lr*slope2
    def predict(self,x):
        return np.dot(x,self.m)+self.b


model = BatchGD(lr=0.02, epochs=10000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
