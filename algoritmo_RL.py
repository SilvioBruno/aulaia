#Algoritmo RL

import numpy as np

class LinearRegression: #Criando a classe
    def __init__(self,x, y): #Construtor
        self.x = np.array(x)
        self.y = np.array(y)
        self.b0 = None #Intercepto
        self.b1 = None #Coeficiente Angular
    def fit(self): #Treinamento, descobrir os parâmetros(b0, b1) de equação linear simples
        xbar = np.mean(self.x) #Média de X
        ybar = np.mean(self.y) #Média de Y
        
        #Coeficiente Angular(b1)
        
        self.b1 = np.sum((self.y - ybar) * (self.x - xbar)) / np.sum((self.x - xbar) **2)
        
        #Intercepto(b0)
        
        self.b0 = ybar - self.b1 * xbar
        y_pred = self.predict(self.x)
        return y_pred
    def predict(self, x_new): #Predição para cada dado de X
        return self.b0 + self.b1 * np.array(x_new)
    def summary(self):
        print(f"Modelo: y ={self.b0} + {self.b1} *x")
        print(f"Intercepto: {self.b0}")
        print(f"Coeficiente Angular: {self.b1}")
        
        
        
#"Só mais um toque do Rei Allan Wa ha boom alakazan"