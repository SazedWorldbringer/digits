from data.loader import load_mnist

X_train, Y_train, y_train, X_test, Y_test, y_test = load_mnist()

print(f"Train images : {X_train.shape} dtype={X_train.dtype}")
print(f"Train labels : {Y_train.shape} dtype={Y_train.dtype}")
print(f"Test images  : {X_test.shape}")
print(f"Test labels  : {Y_test.shape}")

# sanity check first label
print(f"\nFirst training label (raw) : {y_train[0]}")
print(f"First training label (one-hot) : {Y_train[0]}")
print(f"Pixel range: min={X_train.min():.2f} max={X_train.max():.2f}")
