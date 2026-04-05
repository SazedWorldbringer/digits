import numpy as np
import urllib.request
import gzip
import os

DATA_DIR = os.path.join(os.path.dirname(__file__))

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in URLS.items():
        dest = os.path.join(DATA_DIR, name + ".gz")
        if not os.path.exists(dest):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dest)
            print(f"    saved to {dest}")
        else:
            print(f"    {name} already exists, skipping")


def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
    # IDX format: first 16 bytes are header (magic, count, rows, cols)
    # everything after is raw pixel data
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(-1, 784)  # flatten: (60000, 784)
    return images.astype(np.float32)  # normalize to [0, 1]


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
    # first 8 bytes are header
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def one_hot(labels, num_classes=10):
    """
    Convert integer labels to one-hot vectors
    3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    n = len(labels)
    result = np.zeros((n, num_classes), dtype=np.float32)
    result[np.arange(n), labels] = 1.0
    return result


def load_mnist():
    download_mnist()

    train_images = load_images(os.path.join(DATA_DIR, "train_images.gz"))
    train_labels = load_labels(os.path.join(DATA_DIR, "train_labels.gz"))
    test_images = load_images(os.path.join(DATA_DIR, "test_images.gz"))
    test_labels = load_labels(os.path.join(DATA_DIR, "test_labels.gz"))

    return (
        train_images,            # (60000, 784) float32
        one_hot(train_labels),   # (60000, 10) float 32
        # (60000, ) uint8, raw for accuracy calculation
        train_labels,
        test_images,             # (10000, 784)
        one_hot(test_labels),    # (10000, 10)
        test_labels              # (10000, )
    )
