import os.path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from config import PATH_TO_IMAGES,FULLSCALE_FLOW_SHAPE


def write_flo_file(path: str, flow_data: np.ndarray)->None:
    """
    Write optical flow data to a .flo file.

    Args:
        path: Path where the flow file is saved.
        flow_data: Optical flow data of shape (height, width, 2).
    """
    print(f"{path}")
    with open(f"{path}", 'wb') as f:
        # magic number, indicates that its valid flow file
        # .flo file standard header
        header = np.array([80, 73, 69, 72], dtype=np.uint8)
        height, width = flow_data.shape[:2]
        dimensions = np.array([width, height], dtype=np.int32)
        data = flow_data.astype(np.float32).tobytes()
        f.write(header.tobytes())
        f.write(dimensions.tobytes())
        f.write(data)
        print("saved")
def run_flow_estimation_model(model ,sample_idx:int)->np.ndarray:
    """
    :param sample_idx: id of sample from flying chairs to evalute on
    :return: fullscale flow calculated via given model
    """
    img1 =read_ppm_image(f"{PATH_TO_IMAGES}/{sample_idx}_img1.ppm")
    img2 = read_ppm_image(f"{PATH_TO_IMAGES}/{sample_idx}_img2.ppm")
    images_array = [img1.astype(np.float32)[:373, :501, :], img2.astype(np.float32)[:373, :501, :]]
    flow = model.generate_flow(images_array)[0]
    return tf.keras.preprocessing.image.smart_resize(flow, FULLSCALE_FLOW_SHAPE)
def run_flow_estimation_cv2(sample_idx:int) ->np.ndarray:
    img1 = read_ppm_image(f"{PATH_TO_IMAGES}/{sample_idx}_img1.ppm")
    img2 = read_ppm_image(f"{PATH_TO_IMAGES}/{sample_idx}_img2.ppm")
    gray1 = np.array(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    gray2 = np.array(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    flow_farneback = np.array(cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3,
                                                           15, 3, 5, 1.2, 0))
    return flow_farneback
def save_results(output_path:str,file_name:str,flow:np.ndarray,idx:int)->None:
    """
    :param output_path: path to save visualised gt flow, predicted flow , and visualised predicted flow
    :param file_name: name for predicted flow
    :param flow: ndarray containing predicted flow
    :param idx: idx of gt flow
    :return:
    """
    FLOW_FILE_NAME = "{:05d}_flow.flo"
    write_flo_file(output_path + file_name, flow)
    os.system("python -m flowiz " + output_path + file_name)
    os.system(
        "python -m flowiz " + PATH_TO_IMAGES + FLOW_FILE_NAME.format(idx)
        + " --outdir " + output_path)
def read_ppm_image(filename:str)->np.ndarray:
    """

    :param filename:
    :return:
    """
    return np.array(Image.open(filename)).astype(np.float32)
def read_flo_file(filename: str) ->np.ndarray:
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


def flow_to_color(flow: np.ndarray, max_flow=None):
    """Convert an optical flow field to an RGB image.

    Args:
        flow: Optical flow field (numpy array of shape [H, W, 2]).
        max_flow: Optional maximum flow value for normalization.

    Returns:
        flow_color: GRAY image representing the optical flow.
    """

    if max_flow is not None:
        # Normalize flow values by the maximum flow
        flow = np.clip(flow / max_flow, -1, 1)
    else:
        # Normalize flow values by their maximum magnitude
        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        flow = flow / (magnitude[:, :, np.newaxis] + 1e-5)

    # Calculate the angle and magnitude of the flow vectors
    angle = (np.arctan2(-flow[:, :, 1], -flow[:, :, 0]) + np.pi) / (2 * np.pi)
    magnitude = np.clip(np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2), 0, 1)

    # Create an RGB image using the angle and magnitude
    flow_color = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_color[:, :, 0] = (angle * 255).astype(np.uint8)
    flow_color[:, :, 1] = (magnitude * 255).astype(np.uint8)
    flow_color[:, :, 2] = 255

    # Convert from HSV to RGB
    flow_color = cv2.cvtColor(flow_color, cv2.COLOR_BGR2GRAY)

    return flow_color
def calculate_epe(flow_gt : np.ndarray,flow_pred:np.ndarray) -> float:
    """
    :param flow_gt: array containing grand truth optical flow
    :param flow_pred: array containing predicted optical flow
    :return: end-point-error for predicted flow
    """
    du = flow_pred[:, :, 0] - flow_gt[:, :, 0]
    dv = flow_pred[:, :, 1] - flow_gt[:, :, 1]
    endpoint_error = np.sum(np.sqrt(du ** 2 + dv ** 2)) / (flow_gt.shape[0] * flow_gt.shape[1])
    return endpoint_error
def conv2d_leaky_relu(
        layer_input,
        filters: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
        strides: tuple[int, int] = (1, 1),
):
    pad = tf.keras.layers.ZeroPadding2D(padding=padding)(layer_input)
    conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=tf.keras.initializers.he_normal(),
        bias_initializer=tf.keras.initializers.zeros(),
    )(pad)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(conv)


def conv2d_transpose_leaky_relu(
        layer_input,
        filters: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
        strides: tuple[int, int],
):
    pad = tf.keras.layers.ZeroPadding2D(padding=padding)(layer_input)
    conv_trans = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=tf.keras.initializers.he_normal(),
        bias_initializer=tf.keras.initializers.zeros(),

    )(pad)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(conv_trans)


def crop(input, target):
    if input.shape[2:] == target.shape[2:]:
        return input
    else:
        return input[:, :target.shape[1], :target.shape[2], :]
