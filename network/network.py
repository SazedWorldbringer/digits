import numpy as np
from network.activations import sigmoid, sigmoid_prime, softmax


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)

        # zip([784, 128, 10], [128, 10]) gives pairs: (784, 128), (128, 10)
        self.weights = [
            np.random.randn(next_, curr) * 0.1
            for curr, next_ in zip(sizes[:-1], sizes[1:])
        ]
        self.biases = [
            np.zeros((next_, 1))
            for next_ in sizes[1:]
        ]

    def forward(self, X):
        # X shape: (batch-size, 784) -> transpose to (784, batch_size)
        a = X.T

        zs = []  # pre activaiton values, one per layer
        activations = [a]  # post activation values, starts with input

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ a + b  # @ is matrix multiplication
            zs.append(z)

            if i == self.num_layers - 2:  # give probability output for last layer
                a = softmax(z.T).T
            else:  # give sigmoid for layers other than last
                a = sigmoid(z)

            activations.append(a)

        return a, zs, activations

    def loss(self, output, Y):
        # output: (10, batch_size), Y: (batch_size, 10)
        output = output.T  # -> (batch_size, 10)
        return -np.mean(np.sum(Y*np.log(output + 1e-8), axis=1))
