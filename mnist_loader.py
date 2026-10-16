import numpy as np


def load_images(filename):
    """Load MNIST IDX image file."""

    with open(filename, "rb") as f:
        # Read header
        magic = int.from_bytes(f.read(4), byteorder="big")
        num_images = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        cols = int.from_bytes(f.read(4), byteorder="big")

        if magic != 2051:
            raise ValueError("Invalid MNIST image file")

        # Read image data
        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        data = data.reshape(num_images, rows * cols)

        # Convert 0-255 to 0-1
        data = data.astype(np.float64) / 255.0

        # Convert each image to 784 x 1
        data = [
            image.reshape(784, 1)
            for image in data
        ]

        return data


def load_labels(filename):
    """Load MNIST IDX label file."""

    with open(filename, "rb") as f:
        # Read header
        magic = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")

        if magic != 2049:
            raise ValueError("Invalid MNIST label file")

        # Read labels
        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        return labels.tolist()


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0
    in the j'th position and zeroes elsewhere."""

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data():
    """Load MNIST training and test data."""

    training_images = load_images(
        "data/train-images.idx3-ubyte"
    )

    training_labels = load_labels(
        "data/train-labels.idx1-ubyte"
    )

    test_images = load_images(
        "data/t10k-images.idx3-ubyte"
    )

    test_labels = load_labels(
        "data/t10k-labels.idx1-ubyte"
    )

    return (
        training_images,
        training_labels,
        test_images,
        test_labels
    )
