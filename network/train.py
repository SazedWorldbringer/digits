import numpy as np
from network.network import Network


def accuracy(net, X, y_raw):
    output, _, _ = net.forward(X)
    predictions = np.argmax(output, axis=0)  # index of highest confidence
    return np.mean(predictions == y_raw)


def train(X_train, Y_train, y_train, X_test, Y_test, y_test, sizes=[784, 128, 10], epochs=10, batch_size=32, lr=0.01):
    net = Network(sizes)
    n = X_train.shape[0]

    for epoch in range(epochs):
        # shuffle training data each epoch
        idx = np.random.permutation(n)
        X_shuffled = X_train[idx]
        Y_shuffled = Y_train[idx]

        # mini-batch SGD
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            grad_w, grad_b = net.backprop(X_batch, Y_batch)
            net.update(grad_w, grad_b, lr)

        # report progress at end of each epoch
        train_acc = accuracy(net, X_train, y_train)
        test_acc = accuracy(net, X_test, y_test)
        loss = net.loss(net.forward(X_train[:512])[0], Y_train[:512])
        print(f"Epoch {epoch+1:02d} | loss: {loss:.4f} | "
              f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")

    return net
