import numpy as np
import math

x = np.array([34,5,6,4])
random_state = 42
np.random.seed(random_state)
print(np.random.permutation(x)) #embaralhar
tes_size = 0.3 
n_samples = len(x)
n_test = math.ceil(n_samples * tes_size)#arrendonda pra cima
print(n_test)
def train_test_split(X,y,test_size=0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")
    print("Quantidade da amostra", n_samples)
    indices = np.random.permutation(n_samples)
    print("Indices embaralhados", indices) 
    n_test = math.ceil(n_samples * test_size)
    print("Quantidade de amostras para teste", n_test)
    test_indices = indices[:n_test]
    print("Dados de test", test_indices)
    train_indices = indices[n_test:]
    print("Dados de treino",train_indices)
    if X.ndim == 1:
        X_train, X_test = X_train[train_indices], X_test[test_indices]
    else:
        X_train, X_test = X_train[train_indices,:], X_test[test_indices,:]
    y_train, y_test = y[train_indices], y[test_indices]
    X_train, X_test , y_train, y_test

#preste atenção, é o meu cão...