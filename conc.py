import numpy as np
import h5py

def load_dataset():
    """Load the Cat vs Non-Cat dataset from HDF5 files"""
    train_dataset = h5py.File("E:/taha/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("E:/taha/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    # reshape labels to (1, m)
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Load dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Show dataset info
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

print(f"Train set: {m_train} examples")
print(f"Test set: {m_test} examples")

# Flatten and normalize the images
train_x_flatten = train_x_orig.reshape(m_train, -1).T
test_x_flatten = test_x_orig.reshape(m_test, -1).T

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print(f"train_x shape: {train_x.shape}")
print(f"test_x shape: {test_x.shape}")

def initialize(dim):
    """Initialize weights and bias with small random values"""
    w = np.random.randn(dim, 1) * 0.01  
    b = np.random.randn() * 0.01        
    return w, b

def sigmoid(z):
    """Compute sigmoid activation"""
    return 1 / (1 + np.exp(-z))

def propagate(x, y, w, b):
    """
    Forward and backward propagation
    Returns gradients and cost
    """
    m = x.shape[1]
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    cost = -(1/m) * np.sum(y*np.log(a) + (1-y)*np.log(1-a))  # logistic loss

    dw = (1/m) * np.dot(x, (a-y).T)
    db = (1/m) * np.sum(a-y)

    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(x, y, w, b, lr, ni, cp=False):
    """
    Optimize weights and bias by running gradient descent
    """
    costs = []
    for i in range(ni):
        grads, cost = propagate(x, y, w, b)
        dw, db = grads["dw"], grads["db"]

        # gradient descent step
        w -= lr * dw
        b -= lr * db

        if i % 100 == 0:
            costs.append(cost)
            if cp:
                print(f"Cost after iteration {i}: {cost:.6f}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

def predict(w, b, x):
    """
    Predict binary labels for given data
    """
    z = np.dot(w.T, x) + b
    a = sigmoid(z)
    y_prediction = (a > 0.5).astype(int)
    return y_prediction


def model(x_train, y_train, x_test, y_test, lr, ni=5000, cp=False):
    """
    Train logistic regression model and evaluate test accuracy
    """
    w, b = initialize(x_train.shape[0])
    params, grads, costs = optimize(x_train, y_train, w, b, lr, ni, cp)
    w, b = params["w"], params["b"]

    y_predict = predict(w, b, x_test)
    accuracy = 100 - np.mean(np.abs(y_predict - y_test)) * 100
    return round(accuracy, 2)
