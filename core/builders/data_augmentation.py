import tensorflow as tf
import numpy as np
import random
import scipy.ndimage as ndimage


class SequentialDataTransform:
    """
    This class is used for running multiple image and flow transforms sequentialy
    """

    def __init__(self, co_transforms: list[object]):
        """
        :param co_transforms: List of data augmentation class instances which we want to affect
        """
        self.co_transforms = co_transforms
        self.check_if_correct_transforms()

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param images: a pair of images to apply data augmentation on
        :param flow: flow to apply data augmentation on
        :return: tuple of transformed image pair and flows
        """

        for t in self.co_transforms:
            images, flow = t(images, flow)
        return images, flow

    def check_if_correct_transforms(self):
        """
        validates if all passed transforms are objects of transform classes
        :return: True if all elements in list are objects of transform classes
        """
        return True
        for transform in self.co_transforms:
            print(issubclass(type(transform), Transformation))
        if not all(issubclass(type(transform), Transformation)
                   for transform in self.co_transforms):
            raise ValueError("co_transforms should contain instances of transformation classes.")
        return True


class Transformation:
    """
    Base class for image transformations.
    """

    def __init__(self):
        pass
        """
        initialize necessary parameters for transformation
        """

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply transformation to input images and flow vectors.

        :param images: Input images to be transformed.
        :param flow: Associated flow vectors for image transformation.
        :return: Tuple containing transformed images and updated flow vectors.
        """
        pass  # This method should be implemented by subclas


class CenterCrop(Transformation):
    def __init__(self, size: tuple[int, int]):
        """
        :param size: desired output size of cropped image
        """
        super().__init__()
        self.size = size

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param images: a pair of images to apply data augmentation on
        :param flow: flow to apply data augmentation on
        :return: tuple of center cropped image pair and flow
        """
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


class RandomCrop(Transformation):
    def __init__(self, size: tuple[int, int]):
        """
        :param size: desired image size after crop
        """
        super().__init__()
        self.size = size

    def __call__(self, images: list, flow: np.ndarray):
        """
        :param images: a pair of images to apply data augmentation on
        :param flow: flow to apply data augmentation on
        :return: tuple of  cropped image pair and flow
        """
        h, w, _ = images[0].shape
        th, tw = self.size

        if w == tw and h == th:
            return images, flow

        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)
        images[0] = images[0][y1: y1 + th, x1: x1 + tw]
        images[1] = images[1][y1: y1 + th, x1: x1 + tw]
        return images, flow[y1: y1 + th, x1: x1 + tw]


class RandomHorizontalFlip(Transformation):
    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply random horizontal flipping to input images and adjust flow vectors.

        :param images: Input images to be horizontally flipped.
        :param flow: Associated flow vectors for horizontal flipping.
        :return: Tuple containing horizontally flipped images and updated flow vectors.
        """
        if np.random.rand() < 0.5:
            images[0] = np.fliplr(images[0])
            images[1] = np.fliplr(images[1])
            flow = np.fliplr(flow)
            flow[:, :, 0] *= -1
        return images, flow


class RandomVerticalFlip(Transformation):
    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply random vertical flipping to input images and adjust flow vectors.

        :param images: Input images to be vertically flipped.
        :param flow: Associated flow vectors for vertical flipping.
        :return: Tuple containing vertically flipped images and updated flow vectors.
        """
        if np.random.rand() < 0.5:
            images[0] = np.flipud(images[0])
            images[1] = np.flipud(images[1])
            flow = np.flipud(flow)
            flow[:, :, 1] *= -1
        return images, flow


class RandomRotate(Transformation):
    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        super().__init__()
        """
       Initialize RandomRotate transformation parameters.

       :param angle: Maximum angle by which the images will be rotated.
       :param diff_angle: Difference in angle applied to the two images.
       :param order: The order of interpolation for image rotation.
       :param reshape: Whether to reshape the output images after rotation.
       """
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
       Apply random rotation to images and associated flow vectors.

       :param images: Input images to be rotated.
       :param flow: Associated flow vectors for image transformation.
       :return: Tuple containing rotated images and updated flow vectors.
       """
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


class Transformation:
    """
    Base class for image transformations.
    """

    def __call__(self, images: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply transformation to input data.
        This method should be implemented by subclasses based on specific transformation requirements.
        """
        pass


class RandomTranslate(Transformation):
    def __init__(self, translation: tuple[int, int]):
        """
        Initialize RandomTranslate transformation.

        :param translation: Maximum translation in (height, width) to apply to images and flow.
        """
        super().__init__()

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


class Normalize(Transformation):
    def __init__(self, mean: list, std: list):
        """
        Initialize Normalize transformation.

        :param mean: List of mean values for normalization.
        :param std: List of standard deviation values for normalization.
        """
        super().__init__()
        self.mean = mean
        self.std_div = std

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """
        Normalize input array based on mean and standard deviation.

        :param array: Input array to be normalized.
        :return: Normalized array.
        """
        array = (array - self.mean) / self.std_div
        return array.astype(np.float32)


class InverseNormalize(Transformation):
    def __init__(self, mean: list, std: list):
        """
        Initialize InverseNormalize transformation.

        :param mean: List of mean values for inverse normalization.
        :param std: List of standard deviation values for inverse normalization.
        """
        super().__init__()

        self.mean = mean
        self.std_div = std

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """
        Inverse normalize input array based on mean and standard deviation.

        :param array: Input array to be inversely normalized.
        :return: Inverse normalized array.
        """
        array = (array - self.mean) * self.std_div
        return array.astype(np.float32)
