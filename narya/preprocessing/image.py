from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import numpy as np
import cv2
import segmentation_models as sm
from gluoncv.data.transforms.presets.ssd import transform_test
from ..utils.image import (
    np_img_to_torch_img,
    normalize_single_image_torch,
    torch_img_to_np_img,
)

"""Builds the preprocessing function for each model. They all use torch/keras/gluoncv functions depending on the model.
Arguments:
    input_shape: Tuple of integer, the input_shape the model needs to take
Returns:
    preprocessing: function that takes an image as input, and returns the preprocessed image.
"""


def _build_reid_preprocessing(input_shape):
    """Builds the preprocessing function for the Re Identification model.

    """

    def preprocessing(input_img, **kwargs):

        to_normalize = True if np.percentile(input_img, 98) > 1.0 else False

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] / 255.0 if to_normalize else input_img[0] / 1.0
        else:
            image = input_img / 255.0 if to_normalize else input_img / 1.0

        image = cv2.resize(image, (input_shape[1], input_shape[0]))

        image = np_img_to_torch_img(image)
        image = normalize_single_image_torch(
            image, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]
        )
        return image.expand(1, -1, -1, -1)

    return preprocessing


def _build_keypoint_preprocessing(input_shape, backbone):
    """Builds the preprocessing function for the Field Keypoint Detector Model.

    """
    sm_preprocessing = sm.get_preprocessing(backbone)

    def preprocessing(input_img, **kwargs):

        to_normalize = False if np.percentile(input_img, 98) > 1.0 else True

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] * 255.0 if to_normalize else input_img[0] * 1.0
        else:
            image = input_img * 255.0 if to_normalize else input_img * 1.0

        image = cv2.resize(image, input_shape)
        image = sm_preprocessing(image)
        return image

    return preprocessing


def _build_tracking_preprocessing(input_shape):
    """Builds the preprocessing function for the Player/Ball Tracking Model.

    """

    def preprocessing(input_img, **kwargs):

        to_normalize = False if np.percentile(input_img, 98) > 1.0 else True

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] * 255.0 if to_normalize else input_img[0] * 1.0
        else:
            image = input_img * 255.0 if to_normalize else input_img * 1.0

        image = cv2.resize(image, input_shape)
        x, _ = transform_test(mx.nd.array(image), min(input_shape))
        return x

    return preprocessing


def _build_homo_preprocessing(input_shape):
    """Builds the preprocessing function for the Deep Homography estimation Model.

    """

    def preprocessing(input_img, **kwargs):

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0]
        else:
            image = input_img

        image = cv2.resize(image, input_shape)
        image = torch_img_to_np_img(
            normalize_single_image_torch(np_img_to_torch_img(image))
        )
        return image

    return preprocessing
