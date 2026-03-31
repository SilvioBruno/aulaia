import numpy as np
import math


df = np.loadtxt(r"C:\Users\alunok08\Downloads\aulaia-main\mt_cars - mt_cars (2).csv",
                delimiter=",", skiprows=1)

print(df)


X = df[:, 1:10] #características de entrada
y = df[:, 0] #consumo

#hold out

def train_test_split(X,y,test_size=0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")
    n_samples = len(X)
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
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices,:], X[test_indices,:]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test , y_train, y_test



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



class MultipleLinearRegression:
    def __init__(self):
        self.beta_hat = None
    def fit(self, X_train, y_train):
        self.N, self.p = X_train.shape
        X_train = np.column_stack((np.ones((self.N,1)), X_train))
        self.beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    def predict(self, X_new):
        self.N = X_new.shape[0]
        X_new = np.column_stack((np.ones((self.N,1)), X_new))
        return X_new @ self.beta_hat
    
    
model = MultipleLinearRegression()
model.fit(X_train, y_train) #treinamento
y_pred = model.predict(X_test) #predição
print("Real", y_test)
print("Previstos", y_pred)