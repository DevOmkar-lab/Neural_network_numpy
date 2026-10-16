import numpy as np
from network import Network


# Create a neural network
# 2 input neurons
# 3 hidden neurons
# 1 output neuron
net = Network([2, 3, 1])

print("Number of layers:", net.num_layers)
print("Network structure:", net.sizes)

# Create an example input
x = np.array([
    [0.5],
    [0.8]
])

# Run the input through the network
output = net.feedforward(x)

print("Input:")
print(x)

print("Output:")
print(output)
