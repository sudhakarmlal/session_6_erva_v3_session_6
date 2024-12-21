import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

def test_model():
    model = load_model('model_latest_aug.h5')  # Load the latest model
    model.summary()  # Print model summary for parameter count
    print(model.count_params())

    # Test 1: Check number of parameters
    assert model.count_params() < 25000, "Model has more than 25000 parameters"

    # Test 2: Check input shape
    assert model.input_shape == (None, 28, 28, 1), "Model does not accept 28x28 input"

    # Test 3: Check output shape
    assert model.output_shape == (None, 10), "Model does not have 10 outputs"

    # Test 4: Check accuracy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)
    assert accuracy > 0.40, "Model accuracy is less than 40%"

    # Test 5: Check number of layers
    assert len(model.layers) < 30, "Model does not have more than 3 layers"

    # Test 6: Check number of test images
    assert x_test.shape[0] == 10000, "Test set does not have 10000 images"

if __name__ == "__main__":
    test_model()
