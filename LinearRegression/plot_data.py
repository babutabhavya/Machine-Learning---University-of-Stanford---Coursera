import matplotlib.pyplot as plt


def plot_data(X, y):
    print("Plotting Data ...\n")
    plt.scatter(X, y, marker="x", color="r")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()
