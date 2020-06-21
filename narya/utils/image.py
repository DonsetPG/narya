from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from torchvision.transforms import Normalize

from .utils import to_numpy, to_torch


def torch_img_to_np_img(torch_img):
    """Convert a torch image to a numpy image

    Arguments:
        torch_img: Tensor of shape (B,C,H,W) or (C,H,W)
    Returns:
        a np.array of shape (B,H,W,C) or (H,W,C)
    Raises:
        ValueError: If this is not a Torch tensor
    """
    if isinstance(torch_img, np.ndarray):
        return torch_img
    assert isinstance(torch_img, torch.Tensor), "cannot process data type: {0}".format(
        type(torch_img)
    )
    if len(torch_img.shape) == 4 and (
        torch_img.shape[1] == 3 or torch_img.shape[1] == 1
    ):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (
        torch_img.shape[0] == 3 or torch_img.shape[0] == 1
    ):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError("cannot process this image")


def np_img_to_torch_img(np_img):
    """Convert a np image to a torch image

    Arguments:
        np_img: a np.array of shape (B,H,W,C) or (H,W,C)
    Returns:
        a Tensor of shape (B,C,H,W) or (C,H,W)
    Raises:
        ValueError: If this is not a np.array
    """
    if isinstance(np_img, torch.Tensor):
        return np_img
    assert isinstance(np_img, np.ndarray), "cannot process data type: {0}".format(
        type(np_img)
    )
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return to_torch(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return to_torch(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return to_torch(np_img)
    else:
        raise ValueError("cannot process this image")


def normalize_single_image_torch(image, img_mean=None, img_std=None):
    """Normalize a Torch tensor

    Arguments:
        image: Torch Tensor of shape (C,W,H)
        img_mean: List of mean per channel (e.g.: [0.485, 0.456, 0.406])
        img_std: List of std per channel (e.g.: [0.229, 0.224, 0.225])
    Returns:
        image: Torch Tensor of shape (C,W,H), the normalized image
    Raises:
        ValueError: If the shape of the image is not of lenth 3
        ValueError: If the image is not a torch Tensor
    """
    if len(image.shape) != 3:
        raise ValueError(
            "The len(shape) of the image is {}, not 3".format(len(image.shape))
        )
    if isinstance(image, torch.Tensor) == False:
        raise ValueError("The image is not a torch Tensor")
    if img_mean is None and img_std is None:
        img_mean = torch.mean(image, dim=(1, 2)).view(-1, 1, 1)
        img_std = image.contiguous().view(image.size(0), -1).std(-1).view(-1, 1, 1)
        image = (image - img_mean) / img_std
    else:
        image = Normalize(img_mean, img_std, inplace=False)(image)
    return image


def denormalize(x):
    """Scale image to range [0,1]

    Arguments:
        x: np.array, an image
    Returns:
        x: np.array, the scaled image
    Raises:

    """
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
