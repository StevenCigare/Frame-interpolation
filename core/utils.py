
import numpy as np


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


