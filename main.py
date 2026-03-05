import numpy as np
A = np.array([[2,1],[3,4]]) #2x2

A_T = A.T  #Transposta
print(A_T)

A_D = np.linalg.det(A_T) #Determinante

print(A_D)

A_inv = np.linalg.inv(A_T) #Inversa

print(A_inv)



B_inv = np.array([[10,64,2152],
                  [64,522,15114],
                  [2152,15114,701854]])

print(B_inv)