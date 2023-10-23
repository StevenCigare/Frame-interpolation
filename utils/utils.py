import numpy as np
import tensorflow as tf

def write_flo_file(filename, flow_data):
    """
    Write optical flow data to a .flo file.

    Args:
        filename (str): The path to the .flo file.
        flow_data (numpy.ndarray): Optical flow data of shape (height, width, 2).
    """
    with open(filename, 'wb') as f:
        # magic number, indicates that its valid flow file
        # .flo file standard header
        header = np.array([80, 73, 69, 72], dtype=np.uint8)
        height, width = flow_data.shape[:2]
        dimensions = np.array([width, height], dtype=np.int32)
        data = flow_data.astype(np.float32).tobytes()
        f.write(header.tobytes())
        f.write(dimensions.tobytes())
        f.write(data)

def read_flo_file(filename):
    """
    Read optical flow data from a .flo file.

    Args:
        filename (str): The path to the .flo file.

    Returns:
        flow_data (numpy.ndarray): Optical flow data of shape (height, width, 2).
    """
    with open(filename, 'rb') as f:
        # Read the .flo file header
        header = np.fromfile(f, np.uint8, 4)
        if not np.array_equal(header, np.array([80, 73, 69, 72], dtype=np.uint8)):
            raise ValueError("Invalid .flo file format")

        # Read dimensions (width and height)
        dimensions = np.fromfile(f, np.int32, 2)
        width, height = dimensions[0], dimensions[1]

        # Read the flow data
        data = np.fromfile(f, np.float32, width * height * 2)
        flow_data = data.reshape((height, width, 2))

    return flow_data

def conv2d_leaky_relu(
        layer_input,
        filters: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
        strides: tuple[int, int] = (1, 1),
):
    pad = tf.keras.layers.ZeroPadding2D(padding=padding)(layer_input)
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation="relu"
    )(pad)
   # return tf.keras.layers.LeakyReLU(alpha=0.1)(conv)


def conv2d_transpose_leaky_relu(
        layer_input,
        filters: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
        strides: tuple[int, int],
):
    pad = tf.keras.layers.ZeroPadding2D(padding=padding)(layer_input)
    return tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation="relu"
    )(pad)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(conv_trans)


def crop_like(input, target):
    if input.shape[2:] == target.shape[2:]:
        return input
    else:
        return input[:, :target.shape[1], :target.shape[2], :]
