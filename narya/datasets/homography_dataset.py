from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import os
import cv2
import keras
import numpy as np
import keras.backend as K
from lxml import etree
import six
import albumentations as A
import tensorflow as tf

from albumentations import DualTransform

from ..utils.homography import (
    vertical_flip_homo,
    horizontal_flip_homo,
    get_four_corners,
    normalize_homo,
)
from ..preprocessing.image import _build_homo_preprocessing


def new_tf_targets(self):
    """Adds a new object to Albumentations. Allows it to deal not only with:
        * Images
        * Boxes
        * Masks
        * Keypoints
    but also with custom object. In this case: homography.
    We will only need to define custom function ```python3 self.apply_to_object``` for each
    albumentations object we use.
    """
    return {
        "image": self.apply,
        "homo": self.apply_to_homo,
        "mask": self.apply_to_mask,
        "bboxes": self.apply_to_bboxes,
    }


DualTransform.targets = property(new_tf_targets)


class HorizontalFlipWithHomo(A.HorizontalFlip):
    """Class based of albumentations.HorizontalFlip, to allow it to deal with homographies.

    """

    def apply_to_homo(self, homo, **params):
        return horizontal_flip_homo(homo, **params)


class Lambda(A.Lambda):
    """Class based of albumentations.Lambda, to allow it to deal with homographies.

    """

    def apply_to_homo(self, homo, **params):
        return homo


class RandomCropWithHomo(A.RandomCrop):
    """Class based of albumentations.RandomCrop, to allow it to deal with homographies.

    """

    def apply_to_homo(self, homo, **params):
        return homo


class Dataset:
    """Class for an homography dataset. Allows to load pairs of image, homography, and
    apply random transformation to them.

    Arguments:
        images_dir: Path to the folder containing the images, in a '.jpg' format.
        homo_dir: Path to the folder containing the homographies, in a '.npy' format.
                    The homography must have the same name as the image they are linked to.
        augmentation: None if we don't apply random transformation. Else, a function to apply.
        preprocessing: None if we don't apply random preprocessing. Else, a function to apply.
    
    """

    def __init__(self, images_dir, homo_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.homo_fps = [
            os.path.join(homo_dir, image_id.replace(".jpg", "_homo.npy"))
            for image_id in self.ids
        ]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (280, 280))
        homo = np.load(self.homo_fps[i])
        homo = homo[0] if len(homo.shape) > 2 else homo

        # apply augmentations
        temp_homo_0 = homo[0][0]
        if self.augmentation:
            sample = self.augmentation(image=image, homo=homo)
            image, homo = sample["image"], sample["homo"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        homo = normalize_homo(homo)

        if temp_homo_0 != homo[0][0]:
            homo = get_four_corners(homo)[0]
            for i in range(4):
                homo[0][i] = -homo[0][i]
        else:
            homo = get_four_corners(homo)[0]
        return image, homo.flatten()


def get_training_augmentation():
    """Builds random transformations we want to apply to our dataset.

    Arguments:
        
    Returns:
        A albumentation functions to pass our images to.
    Raises:

    """
    train_transform = [
        HorizontalFlipWithHomo(p=0.5),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.augmentations.transforms.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=1,
            shadow_dimension=3,
            always_apply=False,
            p=0.5,
        ),
        A.OneOf([A.RandomBrightness(p=1),], p=0.3,),
        A.OneOf([A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1),], p=0.3,),
        A.OneOf([A.RandomContrast(p=1), A.HueSaturationValue(p=1),], p=0.3,),
        RandomCropWithHomo(height=256, width=256, always_apply=True),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Builds random transformations we want to apply to our dataset.

    Arguments:
        
    Returns:
        A albumentation functions to pass our images to.
    Raises:

    """
    train_transform = [
        HorizontalFlipWithHomo(p=0.5),
        RandomCropWithHomo(height=256, width=256, always_apply=True),
    ]
    return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Arguments:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    Raises:
    """

    _transform = [
        Lambda(name="homo_preprocessing", image=preprocessing_fn),
    ]
    return A.Compose(_transform)


class Dataloder(tf.keras.utils.Sequence):
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

        images = []
        homos = []
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            x_y = self.dataset[j]
            image = x_y[0]
            true_homo = x_y[1]
            images.append(image)
            homos.append(true_homo)

        batch_image = np.array(images)
        batch_true_homo = np.array(homos)

        return (batch_image, batch_true_homo)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


class HomographyDatasetBuilder:
    """Class for an homography dataset. Allows to load pairs of image, homography, and
    apply random transformation to them. Also loads the dataloader you can then pass to a keras model.

    Arguments:
        img_train_dir: Path to the folder containing the training images, in a '.jpg' format.
        img_test_dir: Path to the folder containing the testing images, in a '.jpg' format.
        homo_train_dir: Path to the folder containing the training homographies, in a '.npy' format.
                    The homography must have the same name as the image they are linked to.
        homo_test_dir: Path to the folder containing the testing homographies, in a '.npy' format.
                    The homography must have the same name as the image they are linked to.
        batch_size: Integer number of images in batch.
        preprocess_input: None, or a preprocessing function to apply on the images.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(
        self,
        img_train_dir,
        img_test_dir,
        homo_train_dir,
        homo_test_dir,
        batch_size,
        preprocess_input,
        shuffle=True,
    ):

        self.train_dataset = Dataset(
            img_train_dir,
            homo_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocess_input),
        )

        self.valid_dataset = Dataset(
            img_test_dir,
            homo_test_dir,
            augmentation=get_validation_augmentation(),
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
