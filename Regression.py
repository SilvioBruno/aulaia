import numpy as np


class RegressionM: #construtor
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.beta = None #parametros
        
    def fit(self): #treinamento
        self.X = np.column_stack((np.ones((self.N)), self.X))
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        
    def predict(self, X_new):
        N = X_new.shape[0]
        X_new = np.column_stack((np.ones((N)), X_new))
        return X_new @ self.beta

    
x1 = np.array([2,8,11,10,8,4,2,2,9,8])

x2 = np.array([50,110,120,550,295,200,375,52,100,300])

y = np.array([9.95,24.45,31.75,35,25.02,16.86,14.38,9.6,24.35,27.5])

X = np.column_stack((x1,x2))
    
modelo = RegressionM(X, y)

modelo.fit()
y_pred = modelo.predict(X) #predizer os valores de y
print(y_pred)

def r2_score(y_true, y_prediction):
    numerador = np.sum((y_true - y_prediction)**2)
    denominador = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (numerador/denominador)
    return r2
print(r2_score(y, y_pred))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
#Sem problemas com alcool e jogos...