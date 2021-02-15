from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import random
import os
import cv2
import keras
import numpy as np
import keras.backend as K
from lxml import etree
import six
import albumentations as A

from ..utils.utils import round_clip_0_1
from ..utils.data import _parse_xml_file_keypoints
from ..utils.masks import _build_mask, _flip_keypoint

CLASSES = [str(i) for i in range(29)]# + ["background"]

class Dataset:
    """Class for a keypoint dataset. Allows to load pairs of image, keypoints, and
    apply random transformation to them.

    Arguments:
        images_dir: Path to the folder containing the images, in a '.jpg' format.
        masks_dir: Path to the folder containing the keypoints, in a '.xml' format.
                    The keypoints must have the same name as the image they are linked to.
        classes: List of the classes of keyopints.
        augmentation: None if we don't apply random transformation. Else, a function to apply.
        preprocessing: None if we don't apply random preprocessing. Else, a function to apply.
    
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [
            os.path.join(masks_dir, os.path.splitext(image_id)[0] + ".xml")
            for image_id in self.ids
        ]
                
        # convert str names to class values on masks
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoints = _parse_xml_file_keypoints(self.masks_fps[i])

        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            new_keypoints = {}
            for id_kp, v in six.iteritems(keypoints):
                new_id_kp, x_kp, y_kp = _flip_keypoint(id_kp, min(v[0],image.shape[0]-1), min(v[1],image.shape[1]-1), input_shape = image.shape)
                new_keypoints[new_id_kp] = (x_kp, y_kp)
            keypoints = new_keypoints

        mask = _build_mask(keypoints, mask_shape = (image.shape[0], image.shape[1]))

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    """Builds random transformations we want to apply to our dataset.

    Arguments:
        
    Returns:
        A albumentation functions to pass our images to.
    Raises:

    """
    train_transform = [
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),
        A.OneOf([A.CLAHE(p=1), A.RandomBrightness(p=1), A.RandomGamma(p=1),], p=0.9,),
        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf([A.RandomContrast(p=1), A.HueSaturationValue(p=1),], p=0.9,),
        A.Lambda(mask=round_clip_0_1),
    ]
    return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Arguments:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    Raises:
    """

    _transform = [
        A.Lambda(name="keypoints_preprocessing", image=preprocessing_fn),
    ]
    return A.Compose(_transform)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Arguments:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integer number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


class KeyPointDatasetBuilder:
    """Class for a keypoint dataset. Allows to load pairs of image, keypoints, and
    apply random transformation to them. Also loads the dataloader you can then pass to a keras model.

    Arguments:
        img_train_dir: Path to the folder containing the training images, in a '.jpg' format.
        img_test_dir: Path to the folder containing the testing images, in a '.jpg' format.
        mask_train_dir: Path to the folder containing the training keypoints, in a '.xml' format.
                    The keypoints must have the same name as the image they are linked to.
        mask_test_dir: Path to the folder containing the testing keypoints, in a '.xml' format.
                    The keypoints must have the same name as the image they are linked to.
        batch_size: Integer number of images in batch.
        preprocess_input: None, or a preprocessing function to apply on the images.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(
        self,
        img_train_dir,
        img_test_dir,
        mask_train_dir,
        mask_test_dir,
        batch_size,
        preprocess_input,
        shuffle=True,
    ):

        self.train_dataset = Dataset(
            img_train_dir,
            mask_train_dir,
            classes=CLASSES,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )

        self.valid_dataset = Dataset(
            img_test_dir,
            mask_test_dir,
            classes=CLASSES,
            preprocessing=get_preprocessing(preprocess_input),
        )

        self.train_dataloader = Dataloder(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        self.valid_dataloader = Dataloder(
            self.valid_dataset, batch_size=1, shuffle=False
        )

    def _get_dataset(self):
        return self.train_dataset, self.valid_dataset

    def _get_dataloader(self):
        return self.train_dataloader, self.valid_dataloader


