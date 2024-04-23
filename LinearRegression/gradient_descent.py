import numpy as np
from compute_cost import compute_cost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))
        J_history[iter] = compute_cost(X, y, theta)

    return theta, J_history
