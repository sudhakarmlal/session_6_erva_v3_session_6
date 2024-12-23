
![example workflow](https://github.com/sudhakarmlal/mnist-experiment/actions/workflows/main.yml/badge.svg?event=push)

# MNIST Model Training and Testing


The Model:



class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 26


        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



This project contains two main scripts: `train_model_aug.py` for training a convolutional neural network (CNN) on the MNIST dataset, and `test_model_aug.py` for evaluating the trained model.

## 1. Training the Model (`train_model_aug.py`)

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
python train_model_aug.py

## 3.Testing the Model (`test_model_aug.py`)

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

The `test_model_aug.py` script is used to load the trained model and evaluate its performance on the MNIST test dataset. It includes several assertions to validate the model's architecture and performance.

### Tests Conducted
The following tests are performed in the script:

1. **Check Number of Parameters**: 
   - Asserts that the model has fewer than 20000 parameters

2. **Check Batch Normalization**: 
   - Asserts that the model has used Batch Normalization

3. **Check Droput**: 
   - Asserts that the model has used dropout`.

4. **Check Global Average Pooling**: 
   - Evaluates the model on the Model has used Global Average Pooling.
  


### Usage
To test the model, run the following command:

bash
python test_model_aug.py

## Requirements
Make sure to install the required libraries before running the scripts:

bash
pip install requirements.txt

## Conclusion
This project demonstrates how to train and evaluate a CNN model on the MNIST dataset using Pytorch
