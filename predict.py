from PIL import Image
import numpy as np
import pickle


# Load trained network
with open("trained_network.pkl", "rb") as f:
    net = pickle.load(f)


# Ask for image
filename = input("Enter image filename: ")


# Open image
img = Image.open(filename).convert("L")


# Resize to MNIST size
img = img.resize((28, 28))


# Convert image to NumPy array
img_array = np.array(img)


# Normalize pixel values
img_array = img_array.astype(np.float64) / 255.0


# MNIST uses white digit on black background.
# If your image has black digit on white background,
# invert it.
if img_array.mean() > 0.5:
    img_array = 1.0 - img_array


# Convert to 784 x 1 vector
x = img_array.reshape(784, 1)


# Run through neural network
output = net.feedforward(x)


# Find prediction
prediction = np.argmax(output)


print()
print("==============================")
print("Predicted digit:", prediction)
print("==============================")

print()
print("Output values:")

for digit, value in enumerate(output):
    print(f"{digit}: {value[0]:.4f}")
