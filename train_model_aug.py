import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import datetime

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Image Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,      # Randomly rotate images in the range (degrees)
    width_shift_range=0.1,  # Randomly shift images horizontally
    height_shift_range=0.1, # Randomly shift images vertically
    zoom_range=0.1,         # Randomly zoom into images
    shear_range=0.1,        # Shear transformation
    fill_mode='nearest'     # Fill in new pixels
)

# Fit the generator to the training data
datagen.fit(x_train)

# Plot 100 augmented images
def plot_augmented_images(datagen, x_train):
    plt.figure(figsize=(10, 10))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        img = x_train[np.random.randint(0, x_train.shape[0])]  # Randomly select an image
        img = img.reshape((1, 28, 28, 1))  # Reshape for the generator
        augmented_img = datagen.flow(img).next()[0].reshape(28, 28)  # Generate an augmented image
        plt.imshow(augmented_img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call the function to plot augmented images
plot_augmented_images(datagen, x_train)

# Build the model with fewer than 25,000 parameters
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 16 filters
    layers.BatchNormalization(),  # Added Batch Normalization
    layers.Dropout(0.2),  # Changed to Dropout
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),  # 16 filters
    layers.BatchNormalization(),  # Added Batch Normalization
    layers.Dropout(0.2),  # Changed to Dropout
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (1, 1), activation='relu'),  # Added 1x1 convolution
    layers.BatchNormalization(),  # Added Batch Normalization
    layers.Flatten(),
    layers.Dense(32, activation='relu'),  # 64 units
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adagrad', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 1 epoch with augmented data
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=1, validation_data=(x_test, y_test))

# Save the model with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f'model_latest_aug.h5')
