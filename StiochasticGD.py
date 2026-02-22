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
class StiochasticGD():
    def __init__(self,lr=0.01,epochs=1000):
        self.m=None
        self.b=None
        self.lr=lr
        self.epochs=epochs
    def fit(self,X_train,y_train):
        self.m=np.ones(X_train.shape[1])
        self.b=1
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx=np.random.randint(0,X_train.shape[0])
                
                y_hat=np.dot(self.m,X_train[idx])+self.b
                slope1=-2*(y_train[idx]-y_hat)*X_train[idx]
                slope2=-2*(y_train[idx]-y_hat)

                self.m=self.m-self.lr*slope1
                self.b=self.b-self.lr*slope2
            print(slope1)
    def predict(self,x):
        return np.dot(x,self.m)+self.b


model = StiochasticGD(lr=0.01, epochs=70)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
