# Simple Linear Regression With Gradient Descent from Scratch
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
df = pd.read_csv("ML Datasets/advertising.csv")

def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0  # sum of squared error
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]  # real values
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_derivative_sum = 0
    w_derivative_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_derivative_sum += (y_hat - y)
        w_derivative_sum += (y_hat - y) * X[i]

    new_b = b - (learning_rate * 1 / m * b_derivative_sum)
    new_w = w - (learning_rate * 1 / m * w_derivative_sum)
    return new_b, new_w

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {}, w = {}, mse = {}".format(initial_b, initial_w,
                                                                         cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print("iter={:d}   b={:.2f}   w={:.4f}   mse={:.4f}".format(i, b, w, mse))

    print("After {} iterations b = {}, w = {}, mse = {}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

X = df["radio"]

Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 20000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
