import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Normalize the images to the [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# Expand dimensions to include channel information (needed for Conv2D)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Define different training set sizes to experiment with
training_sizes = [1000, 5000, 10000, 20000, x_train.shape[0]]
results = []
times = []

# Loop through each training set size
for size in training_sizes:
    # Randomly select a subset of the training data
    indices = np.random.choice(x_train.shape[0], size, replace=False)
    x_subset = x_train[indices]
    y_subset = y_train[indices]
    
    # Define a simple CNN model
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=x_train.shape[1:]),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    
    # Compile the model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Train the model and record the training time
    start_time = time.time()
    history = model.fit(x_subset, y_subset, epochs=5, verbose=0)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    
    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results.append(test_acc)
    print(f"Training set size: {size}, Test Accuracy: {test_acc:.4f}, Training Time: {elapsed_time:.2f} seconds")

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_sizes, results, marker='o')
plt.xlabel("Training Set Size")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs. Training Set Size")

plt.subplot(1, 2, 2)
plt.plot(training_sizes, times, marker='o', color='orange')
plt.xlabel("Training Set Size")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time vs. Training Set Size")

plt.tight_layout()
plt.show()
