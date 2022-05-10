import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

file_name = 'data/SCA_example.csv'
test_name = 'testy_example_thesis.csv'

def read_data(filename = file_name, scaling=1):
    S = pd.read_csv(filename).iloc[:, 2:].to_numpy().astype('float64')
    # print(S)
    S[S <= 200] = 0
    return S / scaling

def output_data(pred, filename = file_name):
    res = pd.read_csv(filename).iloc[:, -100:]
    #print(res)
    res = res.iloc[:, -pred.shape[0]:]
    #print(res)
    
    
    
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res.iloc[i, j] = round(pred[j, i])
    res.to_csv(test_name)

def RMSE(X, Y):
    return np.sqrt(
        mean_squared_error(X, Y, multioutput='uniform_average')
    )

def read_related_data(filename= file_name, scaling=5_000):
    S = pd.read_csv(filename).iloc[:, 2:].to_numpy().astype('float64')
    idx = []
    for i, x in enumerate(S):
        if RMSE(x, S[0, :]) < 400:
            idx.append(i)
    return S[idx, :] / scaling

def split_dataset(S, test_size=0.1, shuffle=False, k_th=1):
    X = S[:, :-k_th]
    Y = S[:, k_th:]
    X_train, X_test, Y_train, Y_test = train_test_split(
        np.transpose(X),
        np.transpose(Y),
        test_size=test_size,
        shuffle=shuffle
    )
    return np.transpose(X_train), np.transpose(X_test), np.transpose(Y_train), np.transpose(Y_test)

def normalization_data(S):
    S = S.astype('float64')
    mn = [min(S[i, :]) for i in range(S.shape[0])]
    mx = [max(S[i, :]) for i in range(S.shape[0])]
    for i in range(S.shape[0]):
        S[i] = (S[i] - mn[i]) / (mx[i] - mn[i]) 
    return S, mn, mx

if __name__ == '__main__':
    S = read_related_data()
    print(S[:, -5:])
