import matplotlib.pyplot as plt
from conc import model, load_dataset

# Enable inline plotting (if using Jupyter Notebook)
%matplotlib inline  

# Initialize learning rate and accuracy dictionary
learning_rate = 0.0005
accuracy_results = {}

# Load the dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Dataset info
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

print(f"Train set: {m_train} examples")
print(f"Test set: {m_test} examples")

# Flatten images and normalize pixel values
train_x_flat = train_x_orig.reshape(m_train, -1).T / 255.
test_x_flat = test_x_orig.reshape(m_test, -1).T / 255.

print(f"train_x shape: {train_x_flat.shape}")
print(f"test_x shape: {test_x_flat.shape}")

# Loop through learning rates and record test accuracy
while learning_rate <= 0.01:
    acc = model(train_x_flat, train_y, test_x_flat, test_y, learning_rate)
    accuracy_results[learning_rate] = acc
    learning_rate += 0.0001

# Prepare data for plotting
lrs = list(accuracy_results.keys())
accs = list(accuracy_results.values())

# Plot Accuracy vs Learning Rate
plt.figure(figsize=(8, 6))
plt.plot(lrs, accs, marker='o', linestyle='-', color='b')
plt.title("Accuracy vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()
