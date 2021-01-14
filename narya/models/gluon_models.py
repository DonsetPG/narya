from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import gluoncv as gcv
import numpy as np

from gluoncv.model_zoo import get_model

from ..preprocessing.image import _build_tracking_preprocessing


class TrackerModel:
    """Class for Gluon Models responsible for tracking players and ball on a field. This is only taking care
    of the bouding boxes detections, and not the reidentification.

    Arguments:
        pretrained: Boolean, if the model is loaded pretrained on ImageNet or not
        backbone: String, backbone of the model to load
        input_shape: Tuple, shape of the model's input 
        classes: List of class the model needs to detect
    Call arguments:
        input_img: a np.array of shape input_shape
        split_size: if None, apply the model to the full image. If its an int, the image shape must be divisible by this int.
                    We then split the image to create n smaller images of shape (split_size,split_size), and apply the model
                    to those.
                    We then reconstruct the full images and the full predictions.
    """

    def __init__(
        self,
        pretrained=False,
        backbone="ssd_512_resnet50_v1_coco",
        input_shape=(512, 512),
        classes=["ball", "player"],
        ctx = None
    ):
        if ctx == None:
            try:
                _ = mx.nd.zeros((1,), ctx=mx.gpu(0))
                ctx = [mx.gpu(0)]
            except:
                ctx = [mx.cpu()]

        self.pretrained = pretrained
        self.backbone = backbone
        self.classes = classes
        self.ctx = ctx
        self.model = get_model(self.backbone, pretrained=pretrained, ctx=self.ctx)

        self.model.reset_class(self.classes)
        self.input_shape = input_shape
        self.preprocessing = _build_tracking_preprocessing(input_shape)

    def __call__(self, input_img, split_size=None):

        img_shape = (
            input_img.shape[0] if len(input_img.shape) == 3 else input_img.shape[1]
        )
        ratio = self.input_shape[0] / img_shape
        if split_size:
            cnt = 0
            for i in range(0, img_shape // split_size):
                for j in range(0, img_shape // split_size):
                    x_1, y_1 = i * split_size, j * split_size
                    x_2, y_2 = x_1 + split_size, y_1 + split_size
                    resized_img = input_img[x_1:x_2, y_1:y_2, :]

                    img = self.preprocessing(resized_img)
                    cid, score, bbox = self.model(img)
                    cid, score, bbox = cid.asnumpy(), score.asnumpy(), bbox.asnumpy()
                    full_cid = (
                        cid if cnt == 0 else np.concatenate([full_cid, cid], axis=1)
                    )
                    full_score = (
                        score
                        if cnt == 0
                        else np.concatenate([full_score, score], axis=1)
                    )
                    bbox[:, :, 0] += y_1
                    bbox[:, :, 2] += y_1
                    bbox[:, :, 1] += x_1
                    bbox[:, :, 3] += x_1
                    bbox *= ratio
                    full_bbox = (
                        bbox if cnt == 0 else np.concatenate([full_bbox, bbox], axis=1)
                    )
                    cnt += 1
        else:
            img = self.preprocessing(input_img)
            full_cid, full_score, full_bbox = self.model(img)
        return full_cid, full_score, full_bbox

    def load_weights(self, weights_path):
        try:
            self.model.load_parameters(weights_path, ctx=self.ctx)
            print("Succesfully loaded weights from {}".format(weights_path))
        except:
            orig_weights = "from Imagenet" if self.pretrained else "Randomly"
            print(
                "Could not load weights from {}, weights will be loaded {}".format(
                    weights_path, orig_weights
                )
            )
