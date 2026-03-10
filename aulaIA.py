import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([2,8,11,10,8,4,2,2,9,8])

x2 = np.array([50,110,120,550,295,200,375,52,100,300])

y = np.array([9.95,24.45,31.75,35,25.02,16.86,14.38,9.6,24.35,27.5])


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x1, x2, y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

plt.show()

X =np.column_stack((np.ones(len(x1)),x1, x2))

print(X)

X_T = X.T

print(X_T)

inversa = np.linalg.inv(X_T @ X)

print("A Inversa é \n", inversa)

beta = np.linalg.inv(X_T @ X) @ X_T @ y
print("O que sobra pro Beta ",beta)

#Ajude eles seu traiçoeiro oportunista...