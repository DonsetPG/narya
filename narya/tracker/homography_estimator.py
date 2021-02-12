from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import numpy as np
import tensorflow as tf

from ..models.keras_models import DeepHomoModel
from ..models.keras_models import KeypointDetectorModel
from ..utils.masks import _points_from_mask
from ..utils.homography import get_perspective_transform, compute_homography, warp_point

HOMO_PATH = "https://storage.googleapis.com/narya-bucket-1/models/deep_homo_model.h5"
HOMO_NAME = "deep_homo_model.h5"
HOMO_TOTAR = False

KEYPOINTS_PATH = (
    "https://storage.googleapis.com/narya-bucket-1/models/keypoint_detector.h5"
)
KEYPOINTS_NAME = "keypoint_detector.h5"
KEYPOINTS_TOTAR = False


class HomographyEstimator:
    """Class the homography estimator. Given an image, computes the homography.

    Arguments:
        pretrained: Boolean, if the homography models should be pretrained with our weights or not.
        weights_homo: Path to weight for the homography model
        weights_keypoints: Path to weight for the keypoints model
        shape_in: Shape of the input image
        shape_out: Shape of the ouput image
    Call arguments:
        input_img: np.array
    """

    def __init__(
        self,
        pretrained=True,
        weights_homo=None,
        weights_keypoints=None,
        shape_in=512.0,
        shape_out=320.0,
        keypoint_model_input_shape = (320,320)
    ):

        self.homo_model = DeepHomoModel()
        self.keypoints_model = KeypointDetectorModel(
            backbone="efficientnetb3", num_classes=29, input_shape=keypoint_model_input_shape,
        )

        if pretrained == True:

            checkpoints_homo = tf.keras.utils.get_file(
                HOMO_NAME, HOMO_PATH, HOMO_TOTAR,
            )
            checkpoints_keypoints = tf.keras.utils.get_file(
                KEYPOINTS_NAME, KEYPOINTS_PATH, KEYPOINTS_TOTAR,
            )

            self.homo_model.load_weights(checkpoints_homo)
            self.keypoints_model.load_weights(checkpoints_keypoints)

        elif weights_homo is not None and weights_keypoints is not None:

            self.homo_model.load_weights(weights_homo)
            self.keypoints_model.load_weights(weights_keypoints)

        self.shape_in = shape_in
        self.shape_out = shape_out
        self.ratio = shape_out / shape_in

    def __call__(self, input_img):
        pr_mask = self.keypoints_model(input_img)
        src, dst = _points_from_mask(pr_mask[0])
        if len(src) > 3:
            pred_homo = get_perspective_transform(dst, src)
            method = "cv"
        else:
            corners = self.homo_model(input_img)
            pred_homo = compute_homography(corners)[0]
            method = "torch"
        return pred_homo, method

    def get_field_coordinates(self, bbox, pred_homo, method):
        """Computes the warped coordinates of a bounding box
        Arguments:
            bbox: np.array of shape (4,).
            pred_homo: np.array of shape (3,3)
            method: string in {'cv','torch'}, to precise how the homography was predicted
        Returns:
            dst: np.array, the warped coordinates of the bouding box
        Raises:
        """
        x_1 = int(bbox[0])
        y_1 = int(bbox[1])
        x_2 = int(bbox[2])
        y_2 = int(bbox[3])
        x = (x_1 + x_2) / 2.0 * self.ratio
        y = max(y_1, y_2) * self.ratio
        if method == "cv":
            pts = np.array([float(x), float(y)])
            dst = warp_point(pts, np.linalg.inv(pred_homo), method="cv")
        else:
            pts = np.array([int(x), int(y)])
            dst = warp_point(pts, np.linalg.inv(pred_homo), method="torch")
        return dst
