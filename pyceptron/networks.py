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
        self.wmat = np.random.randn(*self._wshape)
        self.bias = np.random.randn(*self._bshape)
        self.w_iner = np.zeros(self._wshape)
        self.b_iner = np.zeros(self._bshape)

    @property
    def _wshape(self) -> Tuple[int, int]:
        return (self.n_input, self.n_output)

    @property
    def _bshape(self) -> Tuple[int, int]:
        return (self.n_output, 1)

    def gradient_desc(
        self,
        x_inp: NDArray,
        y_out: NDArray,
        iters: int,
        eta: float,
        alpha: float,
        info: bool,
    ) -> None:
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
            # Gradient descent update (with momentum)
            self.wmat += -eta * nabla_w + alpha * self.w_iner
            self.bias += -eta * nabla_b + alpha * self.b_iner
            # Update the inertia terms
            self.w_iner = eta * nabla_w
            self.b_iner = eta * nabla_b
            # Compute and print the loss (MSE)
            error = 0.5 * np.sum(d_error**2)
            if info:
                print(f"Iteration: {i+1:{len(str(iters))}}, Error: {error:.8e}")

    def fit(
        self,
        x_train: NDArray,
        y_train: NDArray,
        iters: int = 1000,
        eta: float = 0.1,
        alpha: float = 0,
        info: bool = True,
    ) -> None:
        x_inp = x_train.T
        y_out = y_train.reshape(self.n_output, -1)
        self.gradient_desc(x_inp, y_out, iters, eta, alpha, info)

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
        self.wmat1 = np.random.randn(*self._wshape1)
        self.bias1 = np.random.randn(*self._bshape1)
        self.wmat2 = np.random.randn(*self._wshape2)
        self.bias2 = np.random.randn(*self._bshape2)
        self.w_iner1 = np.zeros(self._wshape1)
        self.b_iner1 = np.zeros(self._bshape1)
        self.w_iner2 = np.zeros(self._wshape2)
        self.b_iner2 = np.zeros(self._bshape2)

    @property
    def _wshape1(self) -> Tuple[int, int]:
        return (self.n_input, self.n_hidden)

    @property
    def _bshape1(self) -> Tuple[int, int]:
        return (self.n_hidden, 1)

    @property
    def _wshape2(self) -> Tuple[int, int]:
        return (self.n_hidden, self.n_output)

    @property
    def _bshape2(self) -> Tuple[int, int]:
        return (self.n_output, 1)

    def gradient_desc(
        self,
        x_inp: NDArray,
        y_out: NDArray,
        iters: int,
        eta: float,
        alpha: float,
        info: bool,
    ) -> None:
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
            self.wmat2 += -eta * nabla_w2 + alpha * self.w_iner2
            self.bias2 += -eta * nabla_b2 + alpha * self.b_iner2
            self.wmat1 += -eta * nabla_w1 + alpha * self.w_iner1
            self.bias1 += -eta * nabla_b1 + alpha * self.b_iner1
            # Compute and print the loss (MSE)
            error = 0.5 * np.sum(d_error**2)
            if info:
                print(f"Iteration: {i+1:{len(str(iters))}}, Error: {error:.8e}")

    def fit(
        self,
        x_train: NDArray,
        y_train: NDArray,
        iters: int = 1000,
        eta: float = 0.1,
        alpha: float = 0,
        info: bool = True,
    ) -> None:
        x_inp = x_train.T
        y_out = y_train.reshape(self.n_output, -1)
        self.gradient_desc(x_inp, y_out, iters, eta, alpha, info)

    def predict(self, x_test: NDArray) -> NDArray:
        x_test = x_test
        a_one = sigmoid(np.dot(self.wmat1.T, x_test) + self.bias1)
        a_two = sigmoid(np.dot(self.wmat2.T, a_one) + self.bias2)
        return a_two.flatten()


@dataclass
class MultiLayerPerceptron:
    sizes: NDArray | List[int]

    def __post_init__(self):
        self.weights = [np.random.randn(*wshape) for wshape in self._wshapes]
        self.biases = [np.random.randn(*bshape) for bshape in self._bshapes]
        self.w_iners = [np.zeros(wshape) for wshape in self._wshapes]
        self.b_iners = [np.zeros(bshape) for bshape in self._bshapes]

    @property
    def num_layers(self) -> int:
        return len(self.sizes)

    @property
    def _wshapes(self) -> Iterator[Tuple[int, int]]:
        return zip(self.sizes[:-1], self.sizes[1:])

    @property
    def _bshapes(self) -> Iterator[Tuple[int, int]]:
        return zip(self.sizes[1:], np.ones_like(self.sizes[1:]))

    def feedforward(self, x_inp: NDArray) -> Tuple[List[NDArray], List[NDArray]]:
        """Compute the output of the network given an input. The temporary
        values (z and a) are stored for later use in the backpropagation."""
        a_vals, z_vals = [a := x_inp], []
        for w, b in zip(self.weights, self.biases):
            z_vals.append(z := np.dot(w.T, a) + b)
            a_vals.append(a := sigmoid(z))
        return z_vals, a_vals

    def comp_errors(self, y_out: NDArray, z_vals, a_vals) -> List[NDArray]:
        """Compute the error for each layer in the network (deltas)."""
        deltas = [np.zeros_like(zv) for zv in z_vals]
        # Compute the error for the output layer (mse derivative used)
        deltas[-1] = (delta := (a_vals[-1] - y_out) * d_sigmoid(z_vals[-1]))
        # Compute the error for the hidden layers (backpropagation)
        for layer in range(2, self.num_layers):
            aux = np.dot(self.weights[-layer + 1], delta)
            deltas[-layer] = (delta := aux * d_sigmoid(z_vals[-layer]))
        return deltas

    def comp_gradients(self, x_inp: NDArray, y_out: NDArray):
        """Compute the gradients for the weights and biases."""
        z_vals, a_vals = self.feedforward(x_inp)
        deltas = self.comp_errors(y_out, z_vals, a_vals)
        nabla_w = [np.dot(a, d.T) for d, a in zip(deltas, a_vals)]
        nabla_b = [delta.sum(axis=1, keepdims=True) for delta in deltas]
        return nabla_w, nabla_b

    def gradient_desc(
        self,
        x_inp: NDArray,
        y_out: NDArray,
        iters: int,
        eta: float,
        alpha: float,
        info: bool,
    ) -> None:
        "Train the neural network using the gradient descent algorithm."
        for i in range(iters):
            nabla_w, nabla_b = self.comp_gradients(x_inp, y_out)
            # Gradient descent update (with momentum)
            self.weights = [
                wmat - eta * nw + alpha * w_iner
                for wmat, nw, w_iner in zip(self.weights, nabla_w, self.w_iners)
            ]
            self.biases = [
                bias - eta * nb + alpha * b_iner
                for bias, nb, b_iner in zip(self.biases, nabla_b, self.b_iners)
            ]
            # Update the inertia terms
            self.w_iners = [eta * nw for nw in nabla_w]
            self.b_iners = [eta * nb for nb in nabla_b]
            if info:
                error = 0.5 * np.sum((y_out - self.feedforward(x_inp)[1][-1]) ** 2)
                print(f"Iteration: {i+1:{len(str(iters))}}, Error: {error:.8e}")

    def fit(
        self,
        x_train: NDArray,
        y_train: NDArray,
        iters: int = 1000,
        eta: float = 0.1,
        alpha: float = 0,
        info: bool = True,
    ) -> None:
        x_inp = x_train.T
        y_out = y_train.reshape(self.sizes[-1], -1)
        self.gradient_desc(x_inp, y_out, iters, eta, alpha, info)

    def predict(self, x_test: NDArray) -> NDArray:
        x_test = x_test
        pred = self.feedforward(x_test)[1][-1].ravel()
        return pred.flatten()
