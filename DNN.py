import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class MultiplyGate:
    def forward(self, W, X):
        return np.dot(X, W)

    def backward(self, W, X, dZ):
        dW = np.dot(np.transpose(X), dZ)
        dX = np.dot(dZ, np.transpose(W))
        return dW, dX


class addGate:
    def forward(self, b, X):
        return X + b

    def backward(self, dZ):
        # dZ is 2 * 200, but db we need is 1 * num_of_neural
        db = np.dot(np.ones((1, dZ.shape[0]), dtype=np.float64), dZ)
        return db, dZ


class layer:
    def forward(self, X):
        return np.tanh(X)

    def backward(self, X, top_diff):
        output = self.forward(X)
        return (1 - np.square(output)) * top_diff


class softmax:
    def predict(self, X):
        exp_scores = np.exp(X)
        # return the prediction of all the examples
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        probs_yn = probs[range(num_examples), y]
        sum = 0.0
        for i in range(len(y)):
            if y[i] == 0:
                sum += np.log(1 - probs_yn[i])
            else:
                sum += np.log(probs_yn[i])
        return -1.0 / num_examples * sum

    # calc the partial derivative
    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        probs[range(num_examples), y] -= 1
        return probs


class Model:
    def __init__(self, layers_dim):
        self.b = []
        self.W = []
        for i in range(len(layers_dim) - 1):
            self.W.append(np.random.randn(layers_dim[i], layers_dim[i + 1]) / layers_dim[i])
            self.b.append(np.random.rand(layers_dim[i + 1]).reshape(1, layers_dim[i + 1]))

    def calculate_loss(self, X, y):
        mul = MultiplyGate()
        add = addGate()
        tanh = layer()
        output = softmax()

        input = X
        for i in range(len(self.W)):
            input = mul.forward(self.W[i], input)
            input = add.forward(self.b[i], input)
            input = tanh.forward(input)

        return output.loss(input, y)

    def predict(self, X):
        mul = MultiplyGate()
        add = addGate()
        tanh = layer()
        output = softmax()

        input = X
        for i in range(len(self.W)):
            input = mul.forward(self.W[i], input)
            input = add.forward(self.b[i], input)
            input = tanh.forward(input)
        prob = output.predict(input)

        # for every prediction, return the bigger probability(prob is 2 * n, and return 1 * n, 1 is the better axis)
        return np.argmax(prob, axis=1)

    def train(self, X, y, num_passes=20000, epsilon=0.01, reg_lambda=0.01, print_loss=False):
        mul = MultiplyGate()
        add = addGate()
        tanh = layer()
        output = softmax()

        for epoch in range(num_passes):

            input = X
            forward = [(None, None, input)]
            for i in range(len(self.W)):
                m = mul.forward(self.W[i], input)
                a = add.forward(self.b[i], m)
                input = tanh.forward(a)
                forward.append((m, a, input))

            dtanh = output.diff(forward[-1][2], y)
            for i in range(len(forward) - 1, 0, -1):
                dadd = tanh.backward(forward[i][1], dtanh)
                db, dmul = add.backward(dadd)
                dW, dtanh = mul.backward(self.W[i - 1], forward[i - 1][2], dmul)

                dW += reg_lambda * self.W[i - 1]

                self.b[i - 1] -= epsilon * db
                self.W[i - 1] -= epsilon * dW

            if print_loss and epoch % 1000 == 0:
                print("Loss after iteration %i: %f" % (epoch, self.calculate_loss(X, y)))


# visualization
def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)


# generate the data set
np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y)
plt.show()

layers_dim = [2, 5, 2]
model = Model(layers_dim)
model.train(X, y, num_passes=20000, epsilon=0.01, reg_lambda=0.01, print_loss=True)

plot_decision_boundary(lambda x: model.predict(x), X, y)
plt.title("Decision Boundary for hidden layer size n")
plt.show()
