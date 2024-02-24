# In this module, some unit tests are implemented to check
# the correct behavior of the Double Layer Perceptron model.

import numpy as np

from pyceptron import DoubleLayerPerceptron

# Create a univariated Double Layer Perceptron
dlp_univar = DoubleLayerPerceptron(2, 4, 1)

# Createa a mockup dataset
x_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y_train = np.array([1, 1, 1, 0])


def test_dlp_univar_weights_shapes():
    assert dlp_univar.wmat1.shape == (2, 4)
    assert dlp_univar.wmat2.shape == (4, 1)


def test_dlp_univar_biases_shapes():
    assert dlp_univar.bias1.shape == (4, 1)
    assert dlp_univar.bias2.shape == (1, 1)


def test_dlp_univar_w_iners_shapes():
    assert dlp_univar.w_iner1.shape == (2, 4)
    assert dlp_univar.w_iner2.shape == (4, 1)


def test_dlp_univar_b_iners_shapes():
    assert dlp_univar.b_iner1.shape == (4, 1)
    assert dlp_univar.b_iner2.shape == (1, 1)


def test_dlp_univar_fit_execution():
    dlp_univar.fit(x_train, y_train, info=False)


def test_dlp_univar_prediction_shape():
    y_pred = dlp_univar.predict(x_train.T)
    assert y_pred.shape == (4,)
