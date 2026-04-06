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

    def backprop(self, X, Y):
        # X: (batch, 784) Y: (batch, 10)
        batch_size = X.shape[0]

        # forward pass to get all zs and activations
        _, zs, activations = self.forward(X)

        # initialize empty gradient lists, one per layer
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # output layer error
        # activations[-1] is shape (10, batch), Y.T is (10, batch)
        delta = activations[-1] - Y.T

        # gradient for last layer's weights and biases
        grad_w[-1] = (delta @ activations[-2].T) / batch_size
        grad_b[-1] = delta.mean(axis=1, keepdims=True)

        # propagate backwards through hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]  # pre activation at this layer
            sp = sigmoid_prime(z)  # how sensitive was sigmoid here?
            delta = (self.weights[-l+1].T @ delta) * sp  # pull error back

            grad_w[-l] = (delta @ activations[-l-1].T) / batch_size
            grad_b[-l] = delta.mean(axis=1, keepdims=True)

        return grad_w, grad_b

    def update(self, grad_w, grad_b, lr=0.01):
        # nudge every weight opposite to its gradient
        self.weights = [w - lr * gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - lr * gb for b, gb in zip(self.biases, grad_b)]
