"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation.

This is a Python 3 version of the original code.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.

        For example, if the list was [2, 3, 1] then it would be a
        three-layer network, with the first layer containing 2 neurons,
        the second layer 3 neurons, and the third layer 1 neuron.

        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.

        The first layer is assumed to be an input layer, and we won't
        set any biases for those neurons, since biases are only ever
        used in computing the outputs from later layers.
        """

        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [
            np.random.randn(y, 1)
            for y in sizes[1:]
        ]

        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.

        ``training_data`` is a list of tuples ``(x, y)`` representing
        the training inputs and the desired outputs.

        ``epochs`` is the number of times the entire training dataset
        will be used.

        ``mini_batch_size`` is the number of training examples in
        each mini-batch.

        ``eta`` is the learning rate.

        If ``test_data`` is provided, the network will be evaluated
        against the test data after each epoch.
        """

        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        # Python 3: xrange() has been replaced with range()
        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print(
                    "Epoch {0}: {1} / {2}".format(
                        j,
                        self.evaluate(test_data),
                        n_test
                    )
                )
            else:
                print(
                    "Epoch {0} complete".format(j)
                )

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        ``mini_batch`` is a list of tuples ``(x, y)``.
        ``eta`` is the learning rate.
        """

        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [
                nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)
            ]

            nabla_w = [
                nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)
            ]

        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.

        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of
        NumPy arrays, similar to ``self.biases`` and ``self.weights``.
        """

        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        # -------------------------
        # Feedforward
        # -------------------------

        activation = x

        # Store all activations layer by layer
        activations = [x]

        # Store all z vectors layer by layer
        zs = []

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, activation) + b

            zs.append(z)

            activation = sigmoid(z)

            activations.append(activation)

        # -------------------------
        # Backward pass
        # -------------------------

        delta = (
            self.cost_derivative(activations[-1], y)
            * sigmoid_prime(zs[-1])
        )

        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(
            delta,
            activations[-2].transpose()
        )

        # l = 1 means the last layer
        # l = 2 means the second-last layer
        # and so on.

        for l in range(2, self.num_layers):

            z = zs[-l]

            sp = sigmoid_prime(z)

            delta = (
                np.dot(
                    self.weights[-l + 1].transpose(),
                    delta
                )
                * sp
            )

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(
                delta,
                activations[-l - 1].transpose()
            )

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result.

        The neural network's output is assumed to be the index
        of whichever neuron in the final layer has the highest
        activation.
        """

        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]

        return sum(
            int(x == y)
            for (x, y) in test_results
        )

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives
        ∂C_x / ∂a for the output activations.
        """

        return output_activations - y


#### Miscellaneous functions

def sigmoid(z):
    """The sigmoid function."""

    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""

    return sigmoid(z) * (1 - sigmoid(z))
