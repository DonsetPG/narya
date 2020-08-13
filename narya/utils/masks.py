from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np

FLIP_MAPPER = {
    0: 13,
    1: 14,
    2: 15,
    3: 16,
    4: 17,
    5: 18,
    6: 19,
    7: 20,
    8: 21,
    9: 22,
    10: 10,
    11: 11,
    12: 12,
    13: 0,
    14: 1,
    15: 2,
    16: 3,
    17: 4,
    18: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 27,
    24: 28,
    25: 25,
    26: 26,
    27: 23,
    28: 24,
}


def _get_flip_mapper():
    return FLIP_MAPPER


INIT_HOMO_MAPPER = {
    0: [3, 3],
    1: [3, 66],
    2: [51, 65],
    3: [3, 117],
    4: [17, 117],
    5: [3, 203],
    6: [17, 203],
    7: [3, 255],
    8: [51, 254],
    9: [3, 317],
    10: [160, 3],
    11: [160, 160],
    12: [160, 317],
    13: [317, 3],
    14: [317, 66],
    15: [270, 66],
    16: [317, 118],
    17: [304, 118],
    18: [317, 203],
    19: [304, 203],
    20: [317, 255],
    21: [271, 255],
    22: [317, 317],
    23: [51, 128],
    24: [51, 193],
    25: [161, 118],
    26: [161, 203],
    27: [270, 128],
    28: [269, 192],
}


def _get_init_homo_mapper():
    return INIT_HOMO_MAPPER


def _flip_keypoint(id_kp, x_kp, y_kp, input_shape=(320, 320, 3)):
    """Flip the keypoints verticaly, according to the shape of the image

    Arguments:
        id_kep: Integer, the id of the keypoint
        x_kp, y_kp: the x,y coordinates of the keypoint
        input_shapes: Tuple, the shape of the image concerned with the keypoint
    Returns:
        new_id_kp, x_kp, new_y_kp: Tuple of integer with the flipped id and coordinates
    Raises:
        ValueError: If the id_kp is not in the list of Id
        ValueError: If the y coordinates is larger than the input_shape, or smaller than 0
    """

    if id_kp not in FLIP_MAPPER.keys():
        raise ValueError("Keypoint id {} not in the flip mapper".format(id_kp))
    if y_kp < 0 or y_kp > input_shape[0] - 1:
        raise ValueError(
            "y_kp = {}, outside of range [0,{}]".format(y_kp, input_shape[0] - 1)
        )

    new_id_kp = FLIP_MAPPER[id_kp]
    new_y_kp = input_shape[0] - 1 - y_kp

    return (new_id_kp, x_kp, new_y_kp)


def _add_mask(mask, val, x, y):
    """Takes a mask, and add a new segmentation with the value val, around the (x,y) coordinates

    Arguments:
        mask: np.array, the mask
        val: The value to add to the mask 
        x,y: the coordinates of the segmentation to add
    Returns:
        
    Raises:
        
    """
    dir_x = [0, -1, 1]
    dir_y = [0, -1, 1]
    for d_x in dir_x:
        for d_y in dir_y:
            new_x = min(max(x + d_x, 0), mask.shape[0]-1)
            new_y = min(max(y + d_y, 0), mask.shape[1]-1)
            mask[new_x][new_y] = val


def _build_mask(keypoints, mask_shape=(320, 320), nb_of_mask=29):
    """From a dict of keypoints, creates a list of mask with keypoint segmentation

    Arguments:
        keypoints: Dict, mapping each keypoint id to its location
        mask_shape: Shape of the mask to be created
        nb_of_mask: Number of mask to create (= number of different keypoints)
    Returns:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
    Raises:

    """
    mask = np.ones((mask_shape)) * nb_of_mask
    for id_kp, v in six.iteritems(keypoints):
        _add_mask(mask, id_kp, v[0], v[1])
    return mask


def _get_keypoints_from_mask(mask, treshold=0.9):
    """From a list of mask, compute the mapping of each keypoints to their location

    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        keypoints: Dict, mapping each keypoint id to its location
    Raises:
        
    """
    keypoints = {}
    indexes = np.argwhere(mask[:, :, :-1] > treshold)
    for indx in indexes:
        id_kp = indx[2]
        if id_kp in keypoints.keys():
            keypoints[id_kp][0].append(indx[0])
            keypoints[id_kp][1].append(indx[1])
        else:
            keypoints[id_kp] = [[indx[0]], [indx[1]]]

    for id_kp in keypoints.keys():
        mean_x = np.mean(np.array(keypoints[id_kp][0]))
        mean_y = np.mean(np.array(keypoints[id_kp][1]))
        keypoints[id_kp] = [mean_y, mean_x]
    return keypoints

def collinear(p0, p1, p2, epsilon=0.001):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < epsilon

def _points_from_mask(mask, treshold=0.9):
    """From a list of mask, compute src and dst points from the image and the 2D view of the image

    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        src_pts, dst_pts: Location of src and dst related points
    Raises:
        
    """
    list_ids = []
    src_pts, dst_pts = [], []
    available_keypoints = _get_keypoints_from_mask(mask, treshold)
    for id_kp, v in six.iteritems(available_keypoints):
        src_pts.append(v)
        dst_pts.append(INIT_HOMO_MAPPER[id_kp])
        list_ids.append(id_kp)
    src, dst = np.array(src_pts), np.array(dst_pts)

    ### Final test : return nothing if 3 points are colinear and the src has just 4 points 
    test_colinear = False
    if len(src) == 4:
        if collinear(dst_pts[0], dst_pts[1], dst_pts[2]) or collinear(dst_pts[0], dst_pts[1], dst_pts[3]) or collinear(dst_pts[1], dst_pts[2], dst_pts[3]) :
          test_colinear = True
    src = np.array([]) if test_colinear else src
    dst = np.array([]) if test_colinear else dst
    
    return src, dst

