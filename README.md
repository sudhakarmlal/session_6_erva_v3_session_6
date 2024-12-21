
![example workflow](https://github.com/sudhakarmlal/mnist-experiment/actions/workflows/main.yml/badge.svg?event=push)


# MNIST Model Training and Testing

This project contains two main scripts: `train_model.py` for training a convolutional neural network (CNN) on the MNIST dataset, and `test_model.py` for evaluating the trained model.

## 1. Training the Model (`train_model.py`)

### Overview
The `train_model.py` script is responsible for loading the MNIST dataset, building a CNN model, training it, and saving the trained model to a file.

### Key Steps:
- **Load MNIST Dataset**: The script loads the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9).
- **Preprocessing**: The images are reshaped to include a channel dimension and normalized to a range of [0, 1].
- **Model Architecture**: A sequential CNN model is built with:
  - Convolutional layers with ReLU activation
  - Batch normalization layers
  - Dropout layers to prevent overfitting
  - A dense output layer with softmax activation for classification
- **Compilation**: The model is compiled using the Adagrad optimizer and sparse categorical cross-entropy loss.
- **Training**: The model is trained for one epoch on the training dataset.
- **Saving the Model**: The trained model is saved to a file named `model_latest.h5`.

- ## 2.Image Augmentation in `train_model_aug.py`

### Image Augumentation
The `train_model_aug.py` script is responsible for loading the MNIST dataset, applying image augmentation techniques, building a CNN model, training it, and saving the trained model.

### Image Augmentation Techniques
Image augmentation is a technique used to artificially expand the size of a training dataset by creating modified versions of images in the dataset. This helps improve the model's ability to generalize and reduces overfitting. The following augmentation techniques are applied using the `ImageDataGenerator` class from Keras:

- **Rotation**: Randomly rotates images within a specified range (10 degrees in this case).
- **Width Shift**: Randomly shifts images horizontally by a fraction of the total width (10%).
- **Height Shift**: Randomly shifts images vertically by a fraction of the total height (10%).
- **Zoom**: Randomly zooms into images by a factor of 10%.
- **Shear**: Applies a shear transformation to the images.
- **Fill Mode**: Specifies how to fill in new pixels that may appear after transformations (set to 'nearest').

### Usage
To train the model, run the following command:

bash
python train_model.py

## 3.Testing the Model (`test_model.py`)

### Overview
The `test_model.py` script is used to load the trained model and evaluate its performance on the MNIST test dataset.

### Key Steps:
- **Load the Model**: The script loads the trained model from the `model_latest.h5` file.
- **Model Summary**: It prints the model summary and the total number of parameters.
- **Assertions**: Several assertions are made to validate the model:
  - The number of parameters should be less than 25,000.
  - The input shape should be `(None, 28, 28, 1)`.
  - The output shape should be `(None, 10)`.
- **Accuracy Check**: The model is evaluated on the test dataset, and the accuracy should be greater than 95%.

## 4. Tests in `test_model_aug.py`

### Overview
The `test_model_aug.py` script is used to load the trained model and evaluate its performance on the MNIST test dataset. It includes several assertions to validate the model's architecture and performance.

### Tests Conducted
The following tests are performed in the script:

1. **Check Number of Parameters**: 
   - Asserts that the model has fewer than 25,000 parameters.
   - Ensures the model is not overly complex.

2. **Check Input Shape**: 
   - Asserts that the model accepts input shapes of `(None, 28, 28, 1)`.
   - Validates that the model is designed for 28x28 grayscale images.

3. **Check Output Shape**: 
   - Asserts that the model has an output shape of `(None, 10)`.
   - Confirms that the model is set up for 10 classes (digits 0-9).

4. **Check Accuracy**: 
   - Evaluates the model on the test dataset and asserts that the accuracy is greater than 95%.
   - Ensures the model performs well on unseen data.

5. **Check Number of Layers**: 
   - Asserts that the model has more than 3 layers.
   - Validates the complexity of the model architecture.

6. **Check Number of Test Images**: 
   - Asserts that the test dataset contains exactly 10,000 images.
   - Ensures the integrity of the test dataset.

### Usage
To test the model, run the following command:

bash
python test_model_aug.py

## Requirements
Make sure to install the required libraries before running the scripts:

bash
pip install tensorflow numpy

## Conclusion
This project demonstrates how to train and evaluate a CNN model on the MNIST dataset using TensorFlow and Keras.
