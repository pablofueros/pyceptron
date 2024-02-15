# This module contains the implementation of a Single-Layer, Double-
# Layer and Multi-Layer Perceptron (SLP, DLP, MLP respect.) neuronal
# networks using the sigmoid as the activation function and the Mean
# Squared Error (MSE) as the loss function. Created by: Pablo Garcia

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z: NDArray) -> NDArray:
    return sigmoid(z) * (1 - sigmoid(z))


@dataclass
class SingleLayerPerceptron:
    n_input: int
    n_output: int

    def __post_init__(self):
        self.wmat = self.init_wmat()
        self.bias = self.init_bias()

    @property
    def _wshape(self):
        return (self.n_input, self.n_output)

    @property
    def _bshape(self):
        return (self.n_output, 1)

    def init_wmat(self) -> NDArray:
        return np.random.randn(*self._wshape)

    def init_bias(self) -> NDArray:
        return np.random.randn(*self._bshape)

    def gradient_desc(
        self, x_inp: NDArray, y_out: NDArray, iters: int, eta: float, info: bool
    ):
        for i in range(iters):
            # Feed forward
            z_out = np.dot(self.wmat.T, x_inp) + self.bias
            a_out = sigmoid(z_out)
            # Compute the error
            d_error = a_out - y_out
            delta = d_error * d_sigmoid(z_out)
            # Compute the gradients
            nabla_w = np.dot(x_inp, delta.T)
            nabla_b = delta.sum(axis=1, keepdims=True)
            # Gradient descent update
            self.wmat -= eta * nabla_w
            self.bias -= eta * nabla_b
            # Compute and print the loss (MSE)
            error = 0.5 * np.sum(d_error**2)
            if info:
                print(f"Iteration: {i+1:{len(str(iters))}}, Error: {error:.8e}")

    def fit(self, x_train: NDArray, y_train: NDArray, iters=1000, eta=0.1, info=True):
        x_inp = x_train.T
        y_out = y_train.reshape(self.n_output, -1)
        self.gradient_desc(x_inp, y_out, iters, eta, info)

    def predict(self, x_test: NDArray) -> NDArray:
        x_test = x_test
        a_out = sigmoid(np.dot(self.wmat.T, x_test) + self.bias)
        return a_out.flatten()


@dataclass
class DoubleLayerPerceptron:
    n_input: int
    n_hidden: int
    n_output: int

    def __post_init__(self):
        self.wmat1 = self.init_wmat(1)
        self.bias1 = self.init_bias(1)
        self.wmat2 = self.init_wmat(2)
        self.bias2 = self.init_bias(2)

    @property
    def _wshape1(self):
        return (self.n_input, self.n_hidden)

    @property
    def _bshape1(self):
        return (self.n_hidden, 1)

    @property
    def _wshape2(self):
        return (self.n_hidden, self.n_output)

    @property
    def _bshape2(self):
        return (self.n_output, 1)

    def init_wmat(self, layer: int) -> NDArray:
        WSHAPE = {1: self._wshape1, 2: self._wshape2}
        return np.random.randn(*WSHAPE[layer])

    def init_bias(self, layer: int) -> NDArray:
        BSHAPE = {1: self._bshape1, 2: self._bshape2}
        return np.random.randn(*BSHAPE[layer])

    def gradient_desc(
        self, x_inp: NDArray, y_out: NDArray, iters: int, eta: float, info: bool
    ):
        for i in range(iters):
            # Feed forward
            z_one = np.dot(self.wmat1.T, x_inp) + self.bias1
            a_one = sigmoid(z_one)
            z_two = np.dot(self.wmat2.T, a_one) + self.bias2
            a_two = sigmoid(z_two)
            # Compute the errors
            d_error = a_two - y_out
            delta2 = d_error * d_sigmoid(z_two)
            delta1 = np.dot(self.wmat2, delta2) * d_sigmoid(z_one)
            # Compute the gradients
            nabla_w2 = np.dot(a_one, delta2.T)
            nabla_b2 = delta2.sum(axis=1, keepdims=True)
            nabla_w1 = np.dot(x_inp, delta1.T)
            nabla_b1 = delta1.sum(axis=1, keepdims=True)
            # Gradient descent update
            self.wmat2 -= eta * nabla_w2
            self.bias2 -= eta * nabla_b2
            self.wmat1 -= eta * nabla_w1
            self.bias1 -= eta * nabla_b1
            # Compute and print the loss (MSE)
            error = 0.5 * np.sum(d_error**2)
            if info:
                print(f"Iteration: {i+1:{len(str(iters))}}, Error: {error:.8e}")

    def fit(self, x_train: NDArray, y_train: NDArray, iters=1000, eta=0.1, info=True):
        x_inp = x_train.T
        y_out = y_train.reshape(self.n_output, -1)
        self.gradient_desc(x_inp, y_out, iters, eta, info)

    def predict(self, x_test: NDArray) -> NDArray:
        x_test = x_test
        a_one = sigmoid(np.dot(self.wmat1.T, x_test) + self.bias1)
        a_two = sigmoid(np.dot(self.wmat2.T, a_one) + self.bias2)
        return a_two.flatten()


@dataclass
class MultiLayerPerceptron:
    sizes: NDArray | List[int]

    def __post_init__(self):
        self.weights = self.init_weights()
        self.biases = self.init_biases()

    @property
    def num_layers(self) -> int:
        return len(self.sizes)

    @property
    def _wshapes(self) -> Iterator[Tuple[int, int]]:
        return zip(self.sizes[:-1], self.sizes[1:])

    @property
    def _bshapes(self) -> NDArray | List[int]:
        return self.sizes[1:]

    def init_weights(self) -> List[NDArray]:
        return [np.random.randn(y, x) for x, y in self._wshapes]

    def init_biases(self) -> List[NDArray]:
        return [np.random.randn(y, 1) for y in self._bshapes]

    def feedforward(self, x_inp: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
        """Compute the output of the network given an input. The temporary
        values (z and a) are stored for later use in the backpropagation."""
        a_vals, z_vals = [a := x_inp], []
        for w, b in zip(self.weights, self.biases):
            z_vals.append(z := np.dot(w, a) + b)
            a_vals.append(a := sigmoid(z))
        return z_vals, a_vals

    def comp_errors(self, y_out: NDArray, z_vals, a_vals) -> List[NDArray]:
        """Compute the error for each layer in the network (deltas)."""
        deltas = [np.zeros(b.shape) for b in self.biases]
        # Compute the error for the output layer (mse derivative used)
        deltas[-1] = (delta := (a_vals[-1] - y_out) * d_sigmoid(z_vals[-1]))
        # Compute the error for the hidden layers (backpropagation)
        for layer in range(2, self.num_layers):
            aux = np.dot(self.weights[-layer + 1].T, delta)
            deltas[-layer] = (delta := aux * d_sigmoid(z_vals[-layer]))
        return deltas

    def comp_gradients(self, x_inp: NDArray, y_out: NDArray):
        """Compute the gradients for the weights and biases."""
        z_vals, a_vals = self.feedforward(x_inp)
        deltas = self.comp_errors(y_out, z_vals, a_vals)
        nabla_w = [np.dot(d, a.T) for d, a in zip(deltas, a_vals)]
        nabla_b = [delta.sum(axis=1, keepdims=True) for delta in deltas]
        return nabla_w, nabla_b

    def gradient_desc(self, x_inp: NDArray, y_out: NDArray, iters, eta, info):
        "Train the neural network using the gradient descent algorithm."
        for i in range(iters):
            nabla_w, nabla_b = self.comp_gradients(x_inp, y_out)
            self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]
            if info:
                error = 0.5 * np.sum((y_out - self.feedforward(x_inp)[1][-1]) ** 2)
                print(f"Iteration: {i+1:{len(str(iters))}}, Error: {error:.8e}")

    def fit(self, x_train: NDArray, y_train: NDArray, iters=1000, eta=0.1, info=True):
        x_inp = x_train.T
        y_out = y_train.reshape(self.sizes[-1], -1)
        self.gradient_desc(x_inp, y_out, iters, eta, info)

    def predict(self, x_test: NDArray) -> NDArray:
        x_test = x_test
        pred = self.feedforward(x_test)[1][-1].ravel()
        return pred.flatten()
