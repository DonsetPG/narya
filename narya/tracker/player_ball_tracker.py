from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import os
import numpy as np
import tensorflow as tf
import cv2

from ..linker.multitracker import JDETracker
from ..utils.linker import tlwh_to_tlbr
from ..utils.vizualization import (
    plot_tracking,
    rgb_template_to_coord_conv_template,
    merge_template,
)
from ..utils.homography import warp_image
from ..utils.image import np_img_to_torch_img, torch_img_to_np_img
from ..utils.utils import to_torch


class PlayerBallTracker:
    """Class for the Player and Ball tracking. It allows an online tracking and reIdentification of each detected player.
        
    Arguments:
        conf_tresh: Confidence treshold to keep tracked bouding boxes
        track_buffer: Number of frame to keep in memory for tracking reIdentification
        K: Number of boxes to keep at each frames
        frame_rate: -
    """

    def __init__(self, conf_tresh=0.5, track_buffer=30, K=100, frame_rate=30,ctx=None):

        self.frame_rate = frame_rate
        self.tracker = JDETracker(
            conf_thres=conf_tresh, track_buffer=track_buffer, K=K, frame_rate=frame_rate, ctx=ctx
        )

    def get_tracking(
        self,
        imgs,
        results=[],
        begin_frame=0,
        split_size=None,
        verbose=True,
        save_tracking_folder=None,
        template=None,
        frame_to_homo=None,
    ):
        """
        Arguments:
            imgs: List of np.array (images) to track
            results: list of previous results, to resume tracking
            begin_frame: int, starting frame, if you want to resume tracking 
            split_size: if None, apply the tracking model to the full image. If its an int, the image shape must be divisible by this int.
                        We then split the image to create n smaller images of shape (split_size,split_size), and apply the model
                        to those.
                        We then reconstruct the full images and the full predictions.
            verbose: Boolean, to display tracking at each frame or not
            save_tracking_folder: Foler to save the tracking images
            template: Football field, to warp it with the computed homographies on to the saved images
            frame_to_homo: Dict mapping each frame id to a pred_homography and the method used to compute it.
        Returns:
            results: List of results, each result being (frame_id, list of bbox coordiantes, list of bbox id)
            frame_id: Id of the last tracked frame 
        Raises:
        """
        # frame_to_homo: {id: (homo,method)}
        results = results
        frame_id = begin_frame
        for image in imgs:
            resized_image = cv2.resize(image, (512, 512))
            online_targets, ball_bbox = self.tracker.update(
                image, image, split_size=split_size, verbose=verbose
            )
            if ball_bbox is None:
                online_boxs = []
                online_tlwhs = []
                online_ids = []
            else:
                online_tlwhs = [ball_bbox]
                ball_bbox = tlwh_to_tlbr(ball_bbox)
                online_boxs = [ball_bbox]
                online_ids = [-1]

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 100 and not vertical:
                    online_boxs.append(tlwh_to_tlbr(tlwh))
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            results.append((frame_id + 1, online_boxs, online_ids))
            if save_tracking_folder is not None:
                if template is not None and frame_to_homo is not None:
                    pred_homo, method = frame_to_homo[frame_id + 1]
                    conv_template = cv2.resize(
                        rgb_template_to_coord_conv_template(template), (320, 320)
                    )
                    if method == "cv":
                        conv_template = warp_image(
                            conv_template, pred_homo, out_shape=(320, 320)
                        )
                    else:
                        conv_template = warp_image(
                            np_img_to_torch_img(conv_template),
                            to_torch(pred_homo),
                            method="torch",
                        )
                        conv_template = torch_img_to_np_img(conv_template[0])
                    conv_template = cv2.resize(conv_template, (512, 512)).astype(
                        "float32"
                    )
                    resized_image = merge_template(resized_image, conv_template * 255.0)

                online_im = plot_tracking(
                    resized_image, online_tlwhs, online_ids, frame_id=frame_id, fps=1.0
                )
                cv2.imwrite(
                    os.path.join(
                        save_tracking_folder + "test_{:05d}.jpg".format(frame_id)
                    ),
                    online_im,
                )
            frame_id += 1
        return results, frame_id
