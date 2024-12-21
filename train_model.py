# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import datetime

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build the model with fewer than 25,000 parameters
# Build the model with fewer than 25,000 parameters
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 16 filters
    layers.BatchNormalization(),  # Added Batch Normalization
    #layers.Dropout(0.1),  # Changed to Dropout
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),  # 16 filters
    layers.BatchNormalization(),  # Added Batch Normalization
    layers.Dropout(0.1),  # Changed to Dropout
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (1, 1), activation='relu'),  # Added 1x1 convolution
    layers.BatchNormalization(),  # Added Batch Normalization
    layers.Flatten(),
    layers.Dense(32, activation='relu'),  # 64 units
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
print(model.count_params())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 1 epoch
print("ModelLayers:::",model.layers)
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

# Save the model with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f'model_latest.h5')
