from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def isnan(x):
    """Check if x is a number or not.

    Arguments:
        x: an object
    Returns:
        a boolean, == True if x is nan
    Raises:
        
    """
    return x != x


def hasnan(x):
    """Check if a matrix contains nan

    Arguments:
        x: a np.array
    Returns:
        a boolean, == True if x has nan
    Raises:

    """
    return isnan(x).any()


def round_clip_0_1(x, **kwargs):
    """Clip value to [0,1] inside a np.array

    Arguments:
        x: np.array
    Returns:
        np.array with the same shape and clipped value
    Raises:
        
    """
    return x.round().clip(0, 1)


def to_numpy(var):
    """Parse a Torch variable to a numpy array

    Arguments:
        var: torch variable
    Returns:
        a np.array with the same value as var
    Raises:
        
    """
    try:
        return var.numpy()
    except:
        return var.detach().numpy()


def to_torch(np_array):
    """Parse a numpy array to a torch variable

    Arguments:
        np_array: a np.array 
    Returns:
        a torch Var with the same value as the np_array
    Raises:
        
    """
    tensor = torch.from_numpy(np_array).float()
    return torch.autograd.Variable(tensor, requires_grad=False)
