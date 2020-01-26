import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotPoints(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')


def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)


# Activation (sigmoid) function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Output (prediction) formula for one example
def getOutput(features, weights, bias):
    # y_hat_i = sigmoid(Wx+b)
    return sigmoid(np.dot(features, weights) + bias)


# Error (log-loss) formula for one example
def getError(y, output): # y is the label (prediction), either 0 or 1
    return -y * np.log(output) - (1-y)*np.log(1-output)


# Gradient descent step
def updateWeights(x, y, weights, bias, learnrate):
    output = getOutput(x, weights, bias)
    bias += learnrate*(y - output)
    for i in range(len(weights)):
        weights[i] += learnrate*(y - output)*x[i]
    return weights, bias


def train(features, labels, epochs, learnrate):
    # features has size m*n
    errors = []
    m = features.shape[0]  # number of training examples
    n = features.shape[1]  # number of features
    weights = np.random.normal(scale=1 / n**.5, size=n)  # random weights
    bias = 0
    for e in range(epochs):
        for x, y in zip(features, labels):  # x is set of features for one example (size 1*n), y is label
            output = getOutput(x, weights, bias)  # y_i, output of one example (iterates)
            error = getError(y, output)  # Error for one example
            weights, bias = updateWeights(x, y, weights, bias, learnrate)  # update W and b

    plt.title("Solution boundary")
    display(-weights[0] / weights[1], -bias / weights[1], 'black')
    # Plotting the data
    plotPoints(features, labels)
    plt.show()
    return None


data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0,1]]) # features, m*n matrix
y = np.array(data[2]) # labels

np.random.seed(44)
epochs = 100
learnrate = 0.01

train(X, y, epochs, learnrate)
