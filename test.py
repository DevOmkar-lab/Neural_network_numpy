import mnist_loader
import network


print("Loading MNIST data...")

training_images, training_labels, test_images, test_labels = \
    mnist_loader.load_data()


# Convert training data into (x, y) pairs
training_data = [
    (x, mnist_loader.vectorized_result(y))
    for x, y in zip(training_images, training_labels)
]

# Convert test data into (x, y) pairs
test_data = [
    (x, y)
    for x, y in zip(test_images, test_labels)
]


print("Training data:", len(training_data))
print("Test data:", len(test_data))


# Create neural network
net = network.Network([784, 30, 10])

print()
print("Network:", net.sizes)
print("Starting training...")
print()


# Train the network
net.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)

print()
print("Training completed!")
import pickle

with open("trained_network.pkl", "wb") as f:
    pickle.dump(net, f)

print()
print("Training completed!")
print("Trained network saved as trained_network.pkl")
