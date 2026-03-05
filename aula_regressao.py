#Aula de Regressão 


import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

xdata = np.array([1,2,3])
ydata = np.array([1,4,8])

print(xdata,ydata)

plt.scatter(xdata,ydata)
plt.plot(xdata,ydata)
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.title("Gráfico com dados linear")
plt.show()

#Regressão Linear Simples 

xbar = np.mean(xdata) #Média de X
ybar = np.mean(ydata) #Média de Y

#Coeficiente Angular

b1 = np.sum((ydata-ybar)*(xdata-xbar)) / np.sum((xdata-xbar)**2)

#Intercepto

b0 = ybar - b1*xbar
print(b1)
print(b0)

ytest = b0 + b1*xdata
print(ytest)

#R_Score

r_2 = 1 - np.sum((ydata - ytest)**2) / np.sum((ydata - ybar)**2)
print(r_2)