import numpy as np
import matplotlib.pyplot as plt
from warm_up_exercise import warm_up_exercise
from plot_data import plot_data
from compute_cost import compute_cost
from gradient_descent import gradient_descent

warm_up_exercise()


def load_data():
    data = np.loadtxt("ex1data1.txt", delimiter=",")
    X = data[:, 0]
    y = data[:, 1]
    return X, y


X, y = load_data()

# Plot Data
plot_data(X, y)

m = len(y)


# Add a column of ones to X (intercept term)
X = np.column_stack((np.ones(m), X))

# Initiale fitting parameters
theta = np.zeros(2)

# Gradient descent settings
iterations = 1500
alpha = 0.01

# Compute and display initial cost
J = compute_cost(X, y, theta)
print(f"With theta = [0, 0]\nCost computed = {J}")
print("Expected cost value (approx) 32.07\n")

# Gradient descent
theta, _ = gradient_descent(X, y, theta, alpha, iterations)

print("Theta found by gradient descent:")
print(theta)
print("Expected theta values (approx)")
print("-3.6303\n1.1664\n")

# # Plot the linear fit
plt.scatter(X[:, 1], y, marker="x", color="r", label="Training data")
plt.plot(X[:, 1], X.dot(theta), "-", label="Linear regression")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.legend()
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print(f"For population = 35,000, we predict a profit of {predict1*10000}")
predict2 = np.array([1, 7]).dot(theta)
print(f"For population = 70,000, we predict a profit of {predict2*10000}")

# Plot the cost function J(theta)
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(X, y, t)

# Surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_mesh, theta1_mesh, J_vals.T)
ax.set_xlabel("Theta 0")
ax.set_ylabel("Theta 1")
ax.set_zlabel("Cost")

# Contour plot
plt.figure()
plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
plt.xlabel("Theta 0")
plt.ylabel("Theta 1")
plt.plot(theta[0], theta[1], "rx", markersize=10, linewidth=2)
plt.show()
