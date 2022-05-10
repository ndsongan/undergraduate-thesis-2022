from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_preprocessing import read_data, output_data
from sklearn.preprocessing import PolynomialFeatures


DATA_SCALING = 1#20_000

def RMSE(X, Y):
    return np.sqrt(
        mean_squared_error(X, Y, multioutput='uniform_average')
    )


def regression(S, test_size=0.2, mode='linear', degree=1, k_th=1):
    # preprocessing
    n, T = S.shape
    reg = None
    if mode == 'linear':
        reg = LinearRegression()
    elif mode == 'ridge':
        reg = Ridge()
    elif mode == 'lasso':
        reg = Lasso()
    X = S[:, :-k_th]
    Y = S[:, k_th:]
    X_train, X_test, Y_train, Y_test = train_test_split(
        np.transpose(X),
        np.transpose(Y),
        test_size=test_size,
        shuffle=False
    )
    if degree > 1:
        poly = PolynomialFeatures(degree)
        X_train = poly.fit_transform(X_train)
        X_test = poly.fit_transform(X_test)

    # training
    reg.fit(X_train, Y_train)

    print("Train RMSE:", RMSE(reg.predict(X_train), Y_train) * DATA_SCALING)

    # evaluation
    r2_score = reg.score(X_test * DATA_SCALING, Y_test * DATA_SCALING)
    rmse = RMSE(reg.predict(X_test) * DATA_SCALING, Y_test * DATA_SCALING)
    output_data(reg.predict(X_test) * DATA_SCALING)
    return reg, r2_score, rmse

if __name__ == '__main__':
    S = read_data(filename='SCA_example.csv', scaling=DATA_SCALING)[:, -100:]
    reg, r2_score, rmse = regression(
        S,
        test_size=0.2,
        mode='linear'
    )
    print("R2 Score:", r2_score)
    print("RMSE:", rmse)
    # id = random.randint(0, S.shape[1]-2)
    # print("Example:", S[:, id])
    # print("Label:", S[:, id+1])
    # poly = PolynomialFeatures(2)
    # x = np.array([S[:, id]])
    # print("Predict:", np.round_(reg.predict(x)[0]))
