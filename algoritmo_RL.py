#Algoritmo RL

import numpy as np
import plotly.express as px
 

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
        

dados = np.loadtxt(r"C:\Users\alunok08\Desktop\pyaula\slr06 (1) (1).csv",delimiter=",", skiprows=1 )       
print(dados)     
X = dados[:, 0]
Y = dados[:, 1]
fig = px.scatter(x=X, y=Y)

fig.show()

modelo = LinearRegression(X, Y)
modelo.fit()
print("b0 =", modelo.b0)
print("b1 =", modelo.b1)
print(modelo.summary())
x_test = [34, 90, 76, 54]
print(modelo.predict(x_test))

#testar com esses dados
X = np.array([1,2,3])
Y = np.array([1,4,8])

fig2 = px.scatter(x=X, y=Y)

fig2.show()

modelo2 = LinearRegression(X, Y)
modelo2.fit()
print("b0 =", modelo2.b0)
print("b1 =", modelo2.b1)
print(modelo2.summary())
x_test = [34, 90, 76, 54]
print(modelo2.predict(x_test))


#"Só mais um toque do Rei Allan Wa ha boom alakazan"