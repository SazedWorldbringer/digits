import numpy as np
# from network.activations import sigmoid, sigmoid_prime, softmax
from data.loader import load_mnist
from network.train import train
import matplotlib.pyplot as plt

X_train, Y_train, y_train, X_test, Y_test, y_test = load_mnist()
# net = train(X_train, Y_train, Y_train, X_test, Y_test, y_test,
#             sizes=[784, 128, 10], epochs=10, batch_size=32, lr=0.01)
net = train(X_train, Y_train, y_train, X_test, Y_test, y_test,
            sizes=[784, 128, 10], epochs=10, batch_size=32, lr=0.01)


def show_prediction(net, X_test, y_test, idx):
    image = X_test[idx:idx+1]
    output, _, _ = net.forward(image)
    probs = output.flatten()
    pred = np.argmax(probs)
    true = y_test[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(image.reshape(28, 28), cmap='gray')
    ax1.set_title(f"True: {true}  Predicted: {pred}")
    ax1.axis('off')

    ax2.bar(range(10), probs)
    ax2.set_xticks(range(10))
    ax2.set_title("Confidence per digit")
    plt.tight_layout()
    plt.show()


for i in range(10, 20):
    show_prediction(net, X_test, y_test, i)
