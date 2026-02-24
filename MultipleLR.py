import numpy as np
class MLR:
    def __init__(self):
        self.coef=None
        self.intercept=None
    def fit(self,X_train,y_train):
        X_train=np.insert(X_train,0,1,axis=1)
        beta=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.coef=beta[1:]
        self.intercept=beta[0]
    def predict(self,X_test):
        return np.dot(X_test,self.coef)+self.intercept


X = np.array([[1],
              [2],
              [3],
              [4]])

y = np.array([2, 4, 6, 8])

model = MLR()
model.fit(X, y)

print("Coefficient:", model.coef)
print("Intercept:", model.intercept)