import tensorflow as tf
import numpy as np
import random

class Compose:
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, images, flow):
        for t in self.co_transforms:
            images, flow = t(images, flow)
        return images, flow


# will be used for mpisintel dataset
class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h1, w1, _ = images[0].shape
        h2, w2, _ = images[1].shape
        th, tw = self.size

        x1 = (w1 - tw) // 2
        y1 = (h1 - th) // 2
        x2 = (w2 - tw) // 2
        y2 = (h2 - th) // 2

        images[0] = images[0][y1: y1 + th, x1: x1 + tw]
        images[1] = images[1][y2: y2 + th, x2: x2 + tw]
        flow = flow[y1: y1 + th, x1: x1 + tw]
        return images, flow


class RandomCrop:
    def __init__(self, size: tuple[int, int]):
        self.size = size

    def __call__(self, images: list, flow: np.ndarray):
        h, w, _ = images[0].shape
        th, tw = self.size

        if w == tw and h == th:
            return images, flow

        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)
        images[0] = images[0][y1: y1 + th, x1: x1 + tw]
        images[1] = images[1][y1: y1 + th, x1: x1 + tw]
        return images, flow[y1: y1 + th, x1: x1 + tw]


class RandomHorizontalFlip:
    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            images[0] = np.fliplr(images[0])
            images[1] = np.fliplr(images[1])
            flow = np.fliplr(flow)
            flow[:, :, 0] *= -1
        return images, flow


class RandomVerticalFlip:
    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            images[0] = np.flipud(images[0])
            images[1] = np.flipud(images[1])
            flow = np.flipud(flow)
            flow[:, :, 1] *= -1
        return images, flow


class RandomRotate:
    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2
        angle1_rad = angle1 * np.pi / 180
        diff_rad = diff * np.pi / 180

        h, w, _ = flow.shape

        warped_coords = np.mgrid[:w, :h].T + flow
        warped_coords -= np.array([w / 2, h / 2])

        warped_coords_rot = np.zeros_like(flow)

        warped_coords_rot[..., 0] = \
            (np.cos(diff_rad) - 1) * warped_coords[..., 0] + np.sin(diff_rad) * warped_coords[..., 1]

        warped_coords_rot[..., 1] = \
            -np.sin(diff_rad) * warped_coords[..., 0] + (np.cos(diff_rad) - 1) * warped_coords[..., 1]

        flow += warped_coords_rot

        images[0] = ndimage.interpolation.rotate(images[0], angle1, reshape=self.reshape, order=self.order)
        images[1] = ndimage.interpolation.rotate(images[1], angle2, reshape=self.reshape, order=self.order)
        flow = ndimage.interpolation.rotate(flow, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        flow_ = np.copy(flow)
        flow[:, :, 0] = np.cos(angle1_rad) * flow_[:, :, 0] + np.sin(angle1_rad) * flow_[:, :, 1]
        flow[:, :, 1] = -np.sin(angle1_rad) * flow_[:, :, 0] + np.cos(angle1_rad) * flow_[:, :, 1]
        return images, flow


class RandomTranslate:
    def __init__(self, translation: tuple[int, int]):
        self.translation = translation

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = images[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return images, flow
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw), min(w + tw, w), max(0, -tw), min(w - tw, w)
        y1, y2, y3, y4 = max(0, th), min(h + th, h), max(0, -th), min(h - th, h)

        images[0] = images[0][y1:y2, x1:x2]
        images[1] = images[1][y3:y4, x3:x4]
        flow = flow[y1:y2, x1:x2]
        flow[:, :, 0] += tw
        flow[:, :, 1] += th

        return images, flow


class Normalize:
    def __init__(self, mean: list, std: list) -> np.ndarray:
        self.mean = mean
        self.std_div = std

    def __call__(self, array: np.ndarray) -> np.ndarray:
        for elem in array:
            elem = (elem - mean) / std_div
        return array
