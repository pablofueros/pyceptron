# In this module, some unit tests are implemented to check
# the correct behavior of the Sinble Layer Perceptron model.

import numpy as np

from pyceptron import SingleLayerPerceptron

# Create a univariated Single Layer Perceptron
slp_univar = SingleLayerPerceptron(2, 1)

# Createa a mockup dataset
x_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y_train = np.array([1, 1, 1, 0])


def test_slp_univar_weights_shapes():
    assert slp_univar.wmat.shape == (2, 1)


def test_slp_univar_biases_shapes():
    assert slp_univar.bias.shape == (1, 1)


def test_slp_univar_w_iners_shapes():
    assert slp_univar.w_iner.shape == (2, 1)


def test_slp_univar_b_iners_shapes():
    assert slp_univar.b_iner.shape == (1, 1)


def test_slp_univar_fit_execution():
    slp_univar.fit(x_train, y_train, info=False)


def test_slp_univar_prediction_shape():
    y_pred = slp_univar.predict(x_train.T)
    assert y_pred.shape == (4,)
