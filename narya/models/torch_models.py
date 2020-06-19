from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn             as nn

from torchvision            import models

from .torch_layers          import ClassBlock
from ..utils.utils          import to_numpy
from ..preprocessing.image  import _build_reid_preprocessing

#
#
#
# hardcode la fonction de preprocessing
# to_numpy a define
#
#


class ft_net(nn.Module):
    """Class for custom resnet50 torch model. It is the backbone used in the Re Identification model.

    Arguments:
        class_num: Integer, number of class to predict (number of outputs in the last layer)
        droprate: float in [0,1], the Dropout Rate
        stride: Integer, stride parameter in the model
        pretrained: Boolean, if the model is loaded pretrained on ImageNet or not

    """

    def __init__(self, class_num, droprate=0.5, stride=2, pretrained=False):

        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=pretrained)

        self.pretrained = pretrained

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class ReIdModel:
    """Class for Torch Models responsible for the player reIdentification. It takes the image of a player and predict an
    embedding.

    Arguments:
        pretrained: Boolean, if the model is loaded pretrained on ImageNet or not
        input_shape: Tuple, shape of the model's input 
        class_num: Integer, number of class to predict (number of outputs in the last layer)
    Call arguments:
        input_img: a np.array of shape input_shape
    """

    def __init__(self, pretrained=False, input_shape=(256, 128), class_num=751):

        self.pretrained = pretrained
        self.model = ft_net(class_num=class_num, pretrained=pretrained)
        self.input_shape = input_shape
        self.preprocessing = _build_reid_preprocessing(input_shape)

    def __call__(self, input_img):

        img = self.preprocessing(input_img)

        embedding = self.model(img)
        return to_numpy(embedding)

    def load_weights(self, weights_path):
        try:
            self.model.load_state_dict(torch.load(weights_path))
            self.model.eval()
            print("Succesfully loaded weights from {}".format(weights_path))
        except:
            orig_weights = "from Imagenet" if self.pretrained else "Randomly"
            self.model.eval()
            print(
                "Could not load weights from {}, weights will be loaded {}".format(
                    weights_path, orig_weights
                )
            )
