
![example workflow](https://github.com/sudhakarmlal/mnist-experiment/actions/workflows/main.yml/badge.svg?event=push)

# MNIST Model Training and Testing
NoteBook:

https://github.com/sudhakarmlal/session_6_erva_v3_session_6/blob/main/Session6_MNIST_Experiment.ipynb


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

Training Logs:

EPOCH: 0
Loss=0.15141218900680542 Batch_id=468 Accuracy=86.66: 100%|██████████| 469/469 [00:20<00:00, 23.04it/s]
Test set: Average loss: 0.0558, Accuracy: 9843/10000 (98.43%)

EPOCH: 1
Loss=0.04573356732726097 Batch_id=468 Accuracy=97.61: 100%|██████████| 469/469 [00:20<00:00, 23.03it/s]
Test set: Average loss: 0.0390, Accuracy: 9877/10000 (98.77%)

EPOCH: 2
Loss=0.04180360212922096 Batch_id=468 Accuracy=98.12: 100%|██████████| 469/469 [00:22<00:00, 21.01it/s]
Test set: Average loss: 0.0306, Accuracy: 9915/10000 (99.15%)

EPOCH: 3
Loss=0.09601939469575882 Batch_id=468 Accuracy=98.39: 100%|██████████| 469/469 [00:20<00:00, 22.73it/s]
Test set: Average loss: 0.0285, Accuracy: 9909/10000 (99.09%)

EPOCH: 4
Loss=0.043447066098451614 Batch_id=468 Accuracy=98.51: 100%|██████████| 469/469 [00:20<00:00, 22.72it/s]
Test set: Average loss: 0.0314, Accuracy: 9895/10000 (98.95%)

EPOCH: 5
Loss=0.056956950575113297 Batch_id=468 Accuracy=98.66: 100%|██████████| 469/469 [00:21<00:00, 22.09it/s]
Test set: Average loss: 0.0262, Accuracy: 9909/10000 (99.09%)

EPOCH: 6
Loss=0.06073044240474701 Batch_id=468 Accuracy=98.72: 100%|██████████| 469/469 [00:19<00:00, 23.59it/s]
Test set: Average loss: 0.0225, Accuracy: 9928/10000 (99.28%)

EPOCH: 7
Loss=0.017309920862317085 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:20<00:00, 22.91it/s]
Test set: Average loss: 0.0218, Accuracy: 9932/10000 (99.32%)

EPOCH: 8
Loss=0.03483400493860245 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:20<00:00, 22.45it/s]
Test set: Average loss: 0.0236, Accuracy: 9918/10000 (99.18%)

EPOCH: 9
Loss=0.08535811305046082 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:21<00:00, 21.48it/s]
Test set: Average loss: 0.0217, Accuracy: 9923/10000 (99.23%)

EPOCH: 10
Loss=0.03130783140659332 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:20<00:00, 23.30it/s]
Test set: Average loss: 0.0223, Accuracy: 9927/10000 (99.27%)

EPOCH: 11
Loss=0.014949428848922253 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:21<00:00, 21.84it/s]
Test set: Average loss: 0.0190, Accuracy: 9938/10000 (99.38%)

EPOCH: 12
Loss=0.013398057781159878 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:20<00:00, 23.23it/s]
Test set: Average loss: 0.0211, Accuracy: 9931/10000 (99.31%)

EPOCH: 13
Loss=0.013040132820606232 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:20<00:00, 23.23it/s]
Test set: Average loss: 0.0221, Accuracy: 9922/10000 (99.22%)

EPOCH: 14
Loss=0.02502104453742504 Batch_id=468 Accuracy=99.09: 100%|██████████| 469/469 [00:21<00:00, 22.07it/s]
Test set: Average loss: 0.0199, Accuracy: 9933/10000 (99.33%)

EPOCH: 15
Loss=0.057667020708322525 Batch_id=468 Accuracy=99.01: 100%|██████████| 469/469 [00:20<00:00, 23.24it/s]
Test set: Average loss: 0.0193, Accuracy: 9939/10000 (99.39%)

EPOCH: 16
Loss=0.026692770421504974 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:21<00:00, 21.70it/s]
Test set: Average loss: 0.0186, Accuracy: 9934/10000 (99.34%)

EPOCH: 17
Loss=0.008235897868871689 Batch_id=468 Accuracy=99.08: 100%|██████████| 469/469 [00:21<00:00, 21.98it/s]
Test set: Average loss: 0.0200, Accuracy: 9933/10000 (99.33%)

EPOCH: 18
Loss=0.020493581891059875 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:20<00:00, 22.35it/s]
Test set: Average loss: 0.0168, Accuracy: 9943/10000 (99.43%)

EPOCH: 19
Loss=0.0013592600589618087 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:22<00:00, 20.66it/s]
Test set: Average loss: 0.0165, Accuracy: 9943/10000 (99.43%)



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
