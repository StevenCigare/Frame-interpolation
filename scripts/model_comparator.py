from __future__ import annotations

import typing as tp
import cv2
import numpy as np
import tensorflow as tf


class ModelComparator:
    def __init__(self, models: dict[str, list[tp.Any]], images: np.ndarray, flows: np.ndarray):
        """Model comparator initialization.

        Args:
            models: List of model names and paths.
        """
        self.models = self.load_models(models)
        self.images = images
        self.flows = flows
        self._results = dict()

        #delete
        self.best_gunnar = []
        self.best_ours = []
        # delete

    @staticmethod
    def load_models(models: dict[str, list[tp.Any]]) -> dict[str, tp.Any]:
        loaded_models = dict()
        for model_name, data in models.items():
            path, model_class = data
            flow_net = model_class()
            flow_net.create_model()
            flow_net.model.load_weights(path)
            loaded_models.update({model_name: flow_net})

        return loaded_models

    def compare_algorithms(self) -> ModelComparator:
        self._gunnar_farneback()
        return self

    def compare_models(self) -> ModelComparator:
        for model_name, model in self.models.items():
            model_results = []
            for image_pair, flow in zip(self.images, self.flows):
                prediction = model.generate_flow([image_pair[:, :, :3], image_pair[:, :, 3:]])[0]
                model_results.append(float(self._calculate_endpoint_error(prediction, flow)))
            #delete
            self.best_ours = model_results[:]
            #delete
            self._results.update({model_name: np.mean(np.array(model_results))})
        return self

    def _gunnar_farneback(self) -> None:
        results = []
        for image_pair, flow in zip(self.images, self.flows):
            img1 = image_pair[:, :, :3]
            img2 = image_pair[:, :, 3:]
            gray1 = np.array(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
            gray2 = np.array(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
            flow_farneback = np.array(cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0))
            results.append(float(self._calculate_endpoint_error(flow_farneback, flow)))
        # delete
        self.best_gunnar = results[:]
        # delete
        self._results.update({"gunnar_farneback": np.mean(results)})

    def get_results(self) -> dict[str, list[float]]:
        if not self._results:
            raise ValueError("There are no results.")
        return self._results

    # delete
    def get_best_results_indexes(self):
        #difference = np.array(self.best_gunnar) - np.array(self.best_ours)
        indexes = range(20585, len(self.best_ours) + 20585) #20585
        diff_with_indexes = np.transpose(np.array([self.best_ours, indexes]))
        sorted_diff = np.array(sorted(diff_with_indexes, key=lambda x: x[0], reverse=True))[:10]

        print(sorted_diff[:, 1])


    @staticmethod
    def _calculate_endpoint_error(prediction: np.ndarray, flow: np.ndarray):
        return tf.keras.backend.mean(tf.keras.backend.sqrt(
            tf.keras.backend.sum(
                tf.keras.backend.square(
                    tf.keras.preprocessing.image.smart_resize(prediction, flow.shape[:2]) - flow
                ),
                axis=-1, keepdims=True)
            )
        )