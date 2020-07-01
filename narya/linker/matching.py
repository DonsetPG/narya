import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from .kalman_filter import chi2inv95
from ..utils.utils import to_torch, to_numpy

"""

Cloned from https://github.com/Zhongdao/Towards-Realtime-MOT

"""


def linear_assignment(cost_matrix, thresh):
    """Assigns ids based on their cost.
    Arguments:
        cost_matrix: np.array, cost_matrix for pairs of ids
        tresh: float in [0,1], the treshold for id attributions
    Returns:
        matchs: np.array, the list of matches ids
        unmatched_a, unmatched_b: np.array, list of unmatched ids
    Raises:
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """Compute cost based on IoU
    Arguments:
        atlbrs: np.array, list of boxes
        btlbrs: np.array, list of boxes
    Returns:
        ious: np.array, matrix of IoUs between each box
    Raises:
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
    )

    return ious


def iou_distance(atracks, btracks):
    """Compute cost matrix based on IoU for tracks
    Arguments:
        atracks: np.array, list of tracks
        btracks: np.array, list of tracks
    Returns:
        cost_matrix: np.array, matrix of IoU cost for each pair of track
    Raises:
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    """Compute cost based on embedding cosine similarity
    Arguments:
        tracks: list of STrack
        detections: list of BaseTrack
    Returns:
        cost_matrix: np.array, matrix of similarity between each track
    Raises:
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.array([to_numpy(track.curr_feat) for track in detections])
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.array([to_numpy(track.smooth_feat) for track in tracks])
    cost_matrix = np.maximum(
        0.0, cdist(track_features, det_features, metric)
    )  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """Apply a falman-filter and a gating treshold to a cost matrix
    Arguments:
        kf: a KalmanFilter
        cost_matrix: the cost matrix to use
        tracks: a list of STrack
        detections: a list of BaseTrack
    Returns:
        cost_matrix: np.array
    Raises:
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix
