from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import numpy as np
import progressbar
import six
import math 

from scipy.signal import savgol_filter
from scipy import interpolate

from .homography_estimator import HomographyEstimator
from .player_ball_tracker import PlayerBallTracker


class FootballTracker:
    """Class for the full Football Tracker. Given a list of images, it allows to track and id each player as well as the ball.
    It also computes the homography at each given frame, and apply it to each player coordinates. 

    Arguments:
        pretrained: Boolean, if the homography and tracking models should be pretrained with our weights or not.
        weights_homo: Path to weight for the homography model
        weights_keypoints: Path to weight for the keypoints model
        shape_in: Shape of the input image
        shape_out: Shape of the ouput image
        conf_tresh: Confidence treshold to keep tracked bouding boxes
        track_buffer: Number of frame to keep in memory for tracking reIdentification
        K: Number of boxes to keep at each frames
        frame_rate: -
        
    Call arguments:
        imgs: List of np.array (images) to track
        split_size: if None, apply the tracking model to the full image. If its an int, the image shape must be divisible by this int.
                    We then split the image to create n smaller images of shape (split_size,split_size), and apply the model
                    to those.
                    We then reconstruct the full images and the full predictions.
        results: list of previous results, to resume tracking
        begin_frame: int, starting frame, if you want to resume tracking 
        verbose: Boolean, to display tracking at each frame or not
        save_tracking_folder: Foler to save the tracking images
        template: Football field, to warp it with the computed homographies on to the saved images
        skip_homo: List of int. e.g.: [4,10] will not compute homography for frame 4 and 10, and reuse the computed homography
                    at frame 3 and 9.
        enforce_keypoints: Bool. Force the use of the keypoints model. If we can't use it, we skip the frame instead of using the homography model.
        homography_interpolation: Bool. If set to true, missing homography prediction will be computed with an interpolation. If set to false, we simply repeat the 
                                last homography.
        homography_processing: Boo. If set to true, we process the homography estimation with a laplacian filter overtime. 
    """

    def __init__(
        self,
        pretrained=True,
        weights_homo=None,
        weights_keypoints=None,
        shape_in=512.0,
        shape_out=320.0,
        conf_tresh=0.5,
        track_buffer=30,
        K=100,
        frame_rate=30,
        ctx=None
    ):

        self.player_ball_tracker = PlayerBallTracker(
            conf_tresh=conf_tresh, track_buffer=track_buffer, K=K, frame_rate=frame_rate,ctx=ctx
        )

        self.homo_estimator = HomographyEstimator(
            pretrained=pretrained,
            weights_homo=weights_homo,
            weights_keypoints=weights_keypoints,
            shape_in=shape_in,
            shape_out=shape_out,
        )

    def __call__(
        self,
        imgs,
        split_size=None,
        results=[],
        begin_frame=0,
        verbose=True,
        save_tracking_folder=None,
        template=None,
        skip_homo=[],
        enforce_keypoints = False,
        homography_interpolation = False,
        homography_processing = False
    ):
        
        assert enforce_keypoints == homography_interpolation, "We only use homography interpolation with keypoint detection at the moment"

        pred_homo, method = np.ones((3, 3)), "cv"
        
        points, values, methods  = [], [], []

        for indx, input_img in progressbar.progressbar(enumerate(imgs)):
            if indx in skip_homo:
                if homography_interpolation:
                    continue
                else:
                    points.append(indx + 1)
                    values.append(pred_homo)
                    methods.append(method)
            else:
                pred_homo, method = self.homo_estimator(input_img)
                                    
                if (enforce_keypoints and method == 'torch'):
                    if homography_interpolation:
                        continue
                    else:
                        points.append(indx + 1)
                        values.append(values[-1])
                        methods.append(methods[-1])
                else:
                    points.append(indx + 1)
                    values.append(pred_homo)
                    methods.append(method)
    
        points = np.array(points)
        values = np.array(values)
                    
        if homography_interpolation:
            f = interpolate.interp1d(points, values,axis=0,fill_value = 'extrapolate')
            points = np.arange(1,len(imgs)+1)
            values = f(points)
            methods = ["cv" for _ in range(len(imgs))]
            
        if homography_processing:
            values = savgol_filter(values,5,3,axis=0)
                    
        frame_to_homo = {}      
        for indx in range(len(imgs)):
            frame_to_homo[indx + 1] = (values[indx], methods[indx])

        results, frame_id = self.player_ball_tracker.get_tracking(
            imgs,
            results=results,
            begin_frame=begin_frame,
            verbose=verbose,
            split_size=split_size,
            save_tracking_folder=save_tracking_folder,
            template=template,
            frame_to_homo=frame_to_homo,
        )

        last_known_pos = {}
        trajectories = {}
        for result in progressbar.progressbar(results):
            frame, bboxes, id_entities = result[0], result[1], result[2]
            pred_homo, method = frame_to_homo[frame]

            for bbox, id_entity in zip(bboxes, id_entities):
                dst = self.homo_estimator.get_field_coordinates(bbox, pred_homo, method)
                if np.isnan(dst[0]) or np.isnan(dst[1]):
                    if id_entity in last_known_pos.keys():
                        dst = last_known_pos[id_entity]
                    else:
                        dst = None
                if dst is not None:
                    last_known_pos[id_entity] = [dst[0], dst[1]]
                    if id_entity in trajectories.keys():
                        trajectories[id_entity].append((dst[0], dst[1], frame))
                    else:
                        trajectories[id_entity] = [(dst[0], dst[1], frame)]

        return trajectories
