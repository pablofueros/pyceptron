# In this module, some unit tests are implemented to check
# the correct behavior of the Multi Layer Perceptron model.

import numpy as np

from pyceptron import MultiLayerPerceptron

# Create a univariated Multi Layer Perceptron
mlp_univar = MultiLayerPerceptron([2, 4, 4, 1])

# Createa a mockup dataset
x_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y_train = np.array([1, 1, 1, 0])


def test_mlp_univar_weights_shapes():
    assert mlp_univar.weights[0].shape == (2, 4)
    assert mlp_univar.weights[1].shape == (4, 4)
    assert mlp_univar.weights[2].shape == (4, 1)


def test_mlp_univar_biases_shapes():
    assert mlp_univar.biases[0].shape == (4, 1)
    assert mlp_univar.biases[1].shape == (4, 1)
    assert mlp_univar.biases[2].shape == (1, 1)


def test_mlp_univar_w_iners_shapes():
    assert mlp_univar.w_iners[0].shape == (2, 4)
    assert mlp_univar.w_iners[1].shape == (4, 4)
    assert mlp_univar.w_iners[2].shape == (4, 1)


def test_mlp_univar_b_iners_shapes():
    assert mlp_univar.b_iners[0].shape == (4, 1)
    assert mlp_univar.b_iners[1].shape == (4, 1)
    assert mlp_univar.b_iners[2].shape == (1, 1)


def test_mlp_univar_fit_execution():
    mlp_univar.fit(x_train, y_train, info=False)


def test_mlp_univar_prediction_shape():
    y_pred = mlp_univar.predict(x_train.T)
    assert y_pred.shape == (4,)
