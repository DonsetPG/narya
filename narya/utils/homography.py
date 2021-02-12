from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import kornia
import torch
import cv2

from .utils import hasnan, isnan, to_numpy, to_torch
from .image import torch_img_to_np_img, np_img_to_torch_img

"""

Torch warping function cloned from https://github.com/vcg-uvic/sportsfield_release
with some minor modifications

"""


def normalize_homo(h, **kwargs):
    """Normalize an homography by setting the last coefficient to 1.0

    Arguments:
        h: np.array of shape (3,3), the homography
    Returns:
        A np.array of shape (3,3) representing the normalized homography
    Raises:
        
    """
    return h / h[2, 2]


def horizontal_flip_homo(h, **kwargs):
    """Apply a horizontal flip to the homography

    Arguments:
        h: np.array of shape (3,3), the homography
    Returns:
        A np.array of shape (3,3) representing the horizontally flipped homography
    Raises:
        
    """
    flipper = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.matmul(h, flipper)


def vertical_flip_homo(h, **kwargs):
    """Apply a vertical flip to the homography

    Arguments:
        h: np.array of shape (3,3), the homography
    Returns:
        A np.array of shape (3,3) representing the vertically flipped homography
    Raises:
        
    """
    flipper = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    return np.matmul(h, flipper)


def get_perspective_transform_torch(src, dst):
    """Get the homography matrix between src and dst

    Arguments:
        src: Tensor of shape (B,4,2), the four original points per image
        dst: Tensor of shape (B,4,2), the four corresponding points per image
    Returns:
        A tensor of shape (B,3,3), each homography per image
    Raises:

    """
    return kornia.get_perspective_transform(src, dst)


def get_perspective_transform_cv(src, dst):
    """Get the homography matrix between src and dst

    Arguments:
        src: np.array of shape (B,X,2) or (X,2), the X>3 original points per image
        dst: np.array of shape (B,X,2) or (X,2), the X>3 corresponding points per image
    Returns:
        M: np.array of shape (B,3,3) or (3,3), each homography per image
    Raises:

    """
    if len(src.shape) == 2:
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    else:
        M = []
        for src_, dst_ in zip(src, dst):
            M.append(cv2.findHomography(src_, dst_, cv2.RANSAC, 5)[0])
        M = np.array(M)
    return M


def get_perspective_transform(src, dst, method="cv"):
    """Get the homography matrix between src and dst

    Arguments:
        src: Matrix of shape (B,X,2) or (X,2), the X>3 original points per image
        dst: Matrix of shape (B,X,2) or (X,2), the X>3 corresponding points per image
        method: String in {'cv','torch'} to choose which function to use
    Returns:
        M: Matrix of shape (B,3,3) or (3,3), each homography per image
    Raises:

    """
    return (
        get_perspective_transform_cv(src, dst)
        if method == "cv"
        else get_perspective_transform_torch(src, dst)
    )


def warp_image(img, H, out_shape=None, method="cv"):
    """Apply an homography to a Matrix

    Arguments:
        img: Matrix of shape (B,C,H,W) or (C,H,W)
        H: Matrix of shape (B,3,3) or (3,3), the homography
        out_shape: Tuple, the wanted shape of the out image
        method: String in {'cv','torch'} to choose which function to use
    Returns:
        A Matrix of shape (B) x (out_shape) or (B) x (img.shape), the warped image 
    Raises:
        ValueError: If img and H batch sizes are different
    """
    return (
        warp_image_cv(img, H, out_shape=out_shape)
        if method == "cv"
        else warp_image_torch(img, H, out_shape=out_shape)
    )


def warp_image_torch(img, H, out_shape=None):
    """Apply an homography to a torch Tensor

    Arguments:
        img: Tensor of shape (B,C,H,W) or (C,H,W)
        H: Tensor of shape (B,3,3) or (3,3), the homography
        out_shape: Tuple, the wanted shape of the out image
    Returns:
        A Tensor of shape (B) x (out_shape) or (B) x (img.shape), the warped image 
    Raises:
        ValueError: If img and H batch sizes are different
    """
    if out_shape is None:
        out_shape = img.shape[-2:]
    if len(img.shape) < 4:
        img = img[None]
    if len(H.shape) < 3:
        H = H[None]
    if img.shape[0] != H.shape[0]:
        raise ValueError(
            "batch size of images ({}) do not match the batch size of homographies ({})".format(
                img.shape[0], H.shape[0]
            )
        )
    batchsize = img.shape[0]
    # create grid for interpolation (in frame coordinates)

    y, x = torch.meshgrid(
        [
            torch.linspace(-0.5, 0.5, steps=out_shape[-2]),
            torch.linspace(-0.5, 0.5, steps=out_shape[-1]),
        ]
    )
    x = x.to(img.device)
    y = y.to(img.device)
    x, y = x.flatten(), y.flatten()

    # append ones for homogeneous coordinates
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy = xy.repeat([batchsize, 1, 1])  # shape: (B, 3, N)
    # warp points to model coordinates
    xy_warped = torch.matmul(H, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # build grid
    grid = torch.stack(
        [
            x_warped.view(batchsize, *out_shape[-2:]),
            y_warped.view(batchsize, *out_shape[-2:]),
        ],
        dim=-1,
    )

    # sample warped image
    warped_img = torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros"
    )

    if hasnan(warped_img):
        print("nan value in warped image! set to zeros")
        warped_img[isnan(warped_img)] = 0

    return warped_img


def warp_image_cv(img, H, out_shape=None):
    """Apply an homography to a np.array

    Arguments:
        img: np.array of shape (B,H,W,C) or (H,W,C)
        H: Tensor of shape (B,3,3) or (3,3), the homography
        out_shape: Tuple, the wanted shape of the out image
    Returns:
        A np.array of shape (B) x (out_shape) or (B) x (img.shape), the warped image 
    Raises:
        ValueError: If img and H batch sizes are different
    """
    if out_shape is None:
        out_shape = img.shape[-3:-1] if len(img.shape) == 4 else img.shape[:-1]
    if len(img.shape) == 3:
        return cv2.warpPerspective(img, H, dsize=out_shape)
    else:
        if img.shape[0] != H.shape[0]:
            raise ValueError(
                "batch size of images ({}) do not match the batch size of homographies ({})".format(
                    img.shape[0], H.shape[0]
                )
            )
        out_img = []
        for img_, H_ in zip(img, H):
            out_img.append(cv2.warpPerspective(img_, H_, dsize=out_shape))
        return np.array(out_img)


def warp_point(pts, homography, method="cv"):
    return (
        warp_point_cv(pts, homography)
        if method == "cv"
        else warp_point_torch(pts, homography)
    )


def warp_point_cv(pts, homography):
    dst = cv2.perspectiveTransform(np.array(pts).reshape(-1, 1, 2), homography)
    return dst[0][0]


def warp_point_torch(pts, homography, input_shape = (320,320,3)):
    img_test = np.zeros(input_shape)
    dir_ = [0, -1, 1, -2, 2, 3, -3]
    for dir_x in dir_:
        for dir_y in dir_:
            to_add_x = min(max(0, pts[0] + dir_x), 319)
            to_add_y = min(max(0, pts[1] + dir_y), 319)
            for i in range(3):
                img_test[to_add_y, to_add_x, i] = 1.0

    pred_warp = warp_image(
        np_img_to_torch_img(img_test), to_torch(homography), method="torch"
    )
    pred_warp = torch_img_to_np_img(pred_warp[0])
    indx = np.argwhere(pred_warp[:, :, 0] > 0.8)
    x, y = indx[:, 0].mean(), indx[:, 1].mean()
    dst = np.array([y, x])
    return dst


def get_default_corners(batch_size):
    """Get coordinates of the default corners in a soccer field

    Arguments:
        batch_size: Integer, the number of time we need the corners
    Returns:
        orig_corners: a np.array of len(batch_size)
    Raises:
        
    """
    orig_corners = np.array(
        [[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32
    )
    orig_corners = np.tile(orig_corners, (batch_size, 1, 1))
    return orig_corners


def get_corners_from_nn(batch_corners_pred):
    """Gets the corners in the right shape, from a DeepHomoModel

    Arguments:
        batch_corners_pred: np.array of shape (B,8) with the predictions
    Returns:
        corners: np.array of shape (B,4,2) with the corners in the right shape
    Raises:
        
    """
    batch_size = batch_corners_pred.shape[0]
    corners = np.reshape(batch_corners_pred, (-1, 2, 4))
    corners = np.transpose(corners, axes=(0, 2, 1))
    corners = np.reshape(corners, (batch_size, 4, 2))
    return corners


def compute_homography(batch_corners_pred):
    """Compute the homography from the predictions of DeepHomoModel

    Arguments:
        batch_corners_pred: np.array of shape (B,8) with the predictions
    Returns:
        np.array of shape (B,3,3) with the homographies
    Raises:
        
    """
    batch_size = batch_corners_pred.shape[0]
    corners = get_corners_from_nn(batch_corners_pred)
    orig_corners = get_default_corners(batch_size)
    homography = get_perspective_transform_torch(
        to_torch(orig_corners), to_torch(corners)
    )
    return to_numpy(homography)


def get_four_corners(homo_mat):
    """Inverse operation of compute_homography. Gets the 4 corners from an homography.

    Arguments:
        homo_mat: Matrix of shape (B,3,3) or (3,3), homographies
    Returns:
        xy_warped: np.array of shape (B,4,2) with the corners
    Raises:
        ValueError: If the homographies are not of shape (3,3)
    """
    if isinstance(homo_mat, np.ndarray):
        homo_mat = to_torch(homo_mat)

    if homo_mat.shape == (3, 3):
        homo_mat = homo_mat[None]
    if homo_mat.shape[1:] != (3, 3):
        raise ValueError(
            "The shape of the homography is {}, not (3,3)".format(homo_mat.shape[1:])
        )

    canon4pts = to_torch(
        np.array([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32)
    )

    assert canon4pts.shape == (4, 2)
    x, y = canon4pts[:, 0], canon4pts[:, 1]
    xy = torch.stack([x, y, torch.ones_like(x)])
    # warp points to model coordinates
    xy_warped = torch.matmul(homo_mat, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)
    xy_warped = xy_warped / (z_warped + 1e-8)
    xy_warped = to_numpy(xy_warped)
    return xy_warped
