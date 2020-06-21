from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
from mxnet import autograd
from mxnet import gluon

from gluoncv.data import VOCDetection
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.batchify import Tuple, Stack, Pad

CLASSES = ["ball", "player"]

"""
Checking with Mxnet if a GPU is available or not.
"""
try:
    a = mx.nd.zeros((1,), ctx=mx.gpu(0))
    ctx = [mx.gpu(0)]
except:
    ctx = [mx.cpu()]


class VOCFootball(VOCDetection):
    """Class for a tracking dataset. Allows to load pairs of image, bouding boxes, and
    apply random transformation to them. The dataset is base on the gluoncv.data VOC format
    dataset. You can also easily write your own custom dataset if your data are in a COCO format.

    Arguments:
        root: Path to folder storing the dataset
        splits: List of tuples, list of combinations (type,name). e.g: ('foot','train'),('foot','test')
        transform: A function that takes data and label and transforms them. 
                    A transform function for object detection should take label into consideration,
                    because any geometric modification will require label to be modified.
        index_map: In default, the 20 classes are mapped into indices from 0 to 19.
                    We can customize it by providing a str to int dict specifying how to map class names to indices.
                    Use by advanced users only, when you want to swap the orders of class labels.
        preload_label: If True, then parse and load all labels into memory during initialization.
                        It often accelerate speed but require more memory usage.
                        Typical preloaded labels took tens of MB.
                        You only need to disable it when your dataset is extremely large.
    """

    def __init__(
        self, root, splits, transform=None, index_map=None, preload_label=True
    ):

        super(VOCFootball, self).__init__(
            root, splits, transform, index_map, preload_label
        )


def get_dataloader(
    net, train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx
):
    """Loads data from a dataset and returns mini-batches of data, for both the training
    and the validation set.

    Arguments:
        net: the Gluon model you will train, used to generate fake anchors for target generation.
        train_dataset: Training dataset. Note that numpy and mxnet arrays can be directly used as a Dataset.
        val_dataset: Validation dataset. Note that numpy and mxnet arrays can be directly used as a Dataset.
        data_shape: Tuple, the input_shape of the model
        batch_size: Size of mini-batch.
        num_workers: The number of multiprocessing workers to use for data preprocessing.
        ctx: Indicator to the usage of GPU.
    Returns:
        train_loader: Gluon training dataloader
        val_loader: Gluon testing dataloader
    Raises:

    """
    width, height = data_shape

    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx))

    anchors = anchors.as_in_context(mx.cpu())

    batchify_fn = Tuple(
        Stack(), Stack(), Stack()
    )  # stack image, cls_targets, box_targets

    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size,
        True,
        batchify_fn=batchify_fn,
        last_batch="rollover",
        num_workers=num_workers,
    )

    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))

    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size,
        False,
        batchify_fn=val_batchify_fn,
        last_batch="keep",
        num_workers=num_workers,
    )

    return train_loader, val_loader


class TrackingDatasetBuilder:
    """Class for an homography dataset. Allows to load pairs of image, homography, and
    apply random transformation to them. Also loads the dataloader you can then pass to a keras model.

    Arguments:
        dataset_path: Path to folder storing the dataset
        batch_size: Size of mini-batch.
        input_shape: Tuple, the input_shape of the model
        net: the Gluon model you will train, used to generate fake anchors for target generation.
        train_splits: List of tuples, list of combinations (type,name) for training
        test_splits: List of tuples, list of combinations (type,name) for testing
        num_workers: The number of multiprocessing workers to use for data preprocessing.
        
    """

    def __init__(
        self,
        dataset_path,
        batch_size,
        input_shape,
        net,
        train_splits=[("foot", "train"), ("foot", "val")],
        test_splits=[("foot", "test")],
        num_workers=1,
    ):

        self.train_dataset = VOCFootball(root=dataset_path, splits=train_splits)

        self.valid_dataset = VOCFootball(root=dataset_path, splits=test_splits)

        train_loader, val_loader = get_dataloader(
            net,
            self.train_dataset,
            self.valid_dataset,
            input_shape,
            batch_size,
            num_workers,
            ctx,
        )

        self.train_dataloader = train_loader
        self.valid_dataloader = val_loader

    def _get_dataset(self):
        return self.train_dataset, self.valid_dataset

    def _get_dataloader(self):
        return self.train_dataloader, self.valid_dataloader
