from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#
#
#
# hardcode la fonction de preprocessing
# add inputs shape and an assert?
#
#

import mxnet                as mx
import gluoncv              as gcv

from gluoncv.model_zoo      import get_model

from ..preprocessing.image  import _build_tracking_preprocessing


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
    """

    def __init__(
        self,
        pretrained=False,
        backbone="ssd_512_resnet50_v1_coco",
        input_shape=(512, 512),
        classes=["ball", "player"],
    ):

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

    def __call__(self, input_img):

        img = self.preprocessing(input_img)
        cid, score, bbox = self.model(img)
        return cid, score, bbox

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
