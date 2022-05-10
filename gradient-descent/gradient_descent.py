from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from data_preprocessing import output_data
from data_preprocessing import read_data, split_dataset
import random

DATA_SCALING = 10_000

def h(x):
    return np.tanh(x)

def dh(x):
    x = h(x)
    return 1 - x*x

def RMSE(X, Y):
    return np.sqrt(
        mean_squared_error(X, Y, multioutput='uniform_average')
    )

def forward(W, x):
    n = W.shape[0]
    res = np.zeros(n)
    for j in range(n):
        res[j] = np.sum(x * W[:, j])
        res[j] = (h(res[j]) + W[j, n]) * x[j]
    return res

def eval(W, X_test, Y_test, out=False, filename = 'example_SCA.csv'):
    Y_pred = np.array([forward(W, X_test[:, i]) for i in range(X_test.shape[1])])
    if out:
        output_data(Y_pred * DATA_SCALING)
        #np.savetxt(filename, np.round(Y_pred), delimiter = ',')
    Y_pred = np.transpose(Y_pred) * DATA_SCALING
    r2 = r2_score(Y_test * DATA_SCALING, Y_pred)
    rmse = RMSE(Y_test * DATA_SCALING, Y_pred)
    return r2, rmse

def gradient_descent(S, alpha, split_alg, iteration=1000, test_size=0.1):
    # preprocessing
    #X_train, X_test, Y_train, Y_test = split_alg(S, test_size)
    X_train = S
    X_test = S
    Y_train = S
    Y_test = S
    print(S)
    print(X_train)
    #print(X_test)
    #print(Y_train)
    #print(Y_test)
    n, T = X_train.shape
    print("n = " + str(n))
    W = np.array([[0.6233599,  0.91830491, 0.96521638, 0.30289238], 
         [0.23904069, 0.37325159, 0.64574142, 0.94691977], 
         [0.74219297, 0.19658526, 0.2458533,  0.32731402]]) #np.random.rand(n, n+1) #- 0.5 * 2
    print("before")
    print(W)
    finish_percentage = 1
    iter_score = []

    # training
    print("Training:")
    for iter in range(iteration):
        if (iter + 1) / iteration >= finish_percentage / 10:
            print("{}%".format(finish_percentage * 10))
            finish_percentage += 1
            
            #print(iter_score[-1])
        for t in range(T):
            for j in range(n):
                yj = np.sum(W[:n, j] * X_train[:n, t])
                temp = yj
                print("before yj")
                print(yj)
                yj = (h(yj) + W[j, n]) * X_train[j, t]
                print("after yj")
                print(yj)
                for i in range(n):
                    delta = alpha * (yj - Y_train[j, t]) * X_train[i, t] * X_train[j, t] * dh(temp)
                    print("delta = " + str(delta))
                    W[i, j] -= delta
                delta = alpha * (yj - Y_train[j, t]) * X_train[j, t]
                W[j, n] -= delta
        if (iter + 1) % 1 == 0:
            iter_score.append(eval(W, X_train, Y_train))
    print("after")
    print(W)
    # evaluation
    print("Train eval:", eval(W, X_train, Y_train, True, 'train_result_SCA_example.csv'))
    r2, rmse = eval(W, X_test, Y_test, True, 'test_result_SCA_example')
    return W, r2, rmse, iter_score

if __name__ == '__main__':
    S = read_data(filename='SCA_example.csv', scaling=DATA_SCALING)[:, -100:]
    # S = read_data(filename='data/dataset.csv')
    #print(S)
    split_alg = lambda S, test_size: split_dataset(S, test_size)
    W, r2, rmse, iter_score = gradient_descent(S, 0.01, split_alg, iteration= 2)
    #np.save("weight_SCA.npy", W)
    iter_score = np.array([x[1] for x in iter_score])
    #np.save("iter_SCA.npy", iter_score)
    print("R2: ", r2)
    print("RMSE: ", rmse)
    id = random.randint(0, S.shape[1]-2)
    print("Example:", (S[:, id] * DATA_SCALING).astype('int64'))
    print("Label:", (S[:, id+1] * DATA_SCALING).astype('int64'))
    print("Predict:", (forward(W, S[:, id]) * DATA_SCALING).astype('int64'))
    
    
    # weights = np.load('weight_example.npy')
    # print(weights)
