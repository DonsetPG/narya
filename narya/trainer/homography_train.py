from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import argparse
import segmentation_models as sm
import tensorflow as tf
import keras
import os

from narya.models.keras_models import DeepHomoModel
from narya.datasets.homography_dataset import HomographyDatasetBuilder

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--data_dir", default="dataset/", type=str)
parser.add_argument("--x_train_dir", default="train_img", type=str)
parser.add_argument("--y_train_dir", default="train_homo", type=str)
parser.add_argument("--x_test_dir", default="test_img", type=str)
parser.add_argument("--y_test_dir", default="test_homo", type=str)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--weights", default=None, type=str)
opt = parser.parse_args()

"""
Launching with:
```
python3 narya/trainer/homography_train.py --data_dir 'dataset/ \
                                        --x_train_dir 'train_img' \
                                        --y_train_dir 'train_homo' \
                                        --x_test_dir 'test_img' \
                                        --y_test_dir 'test_homo' \
                                        --batch_size 2 \
                                        --lr 0.0001 \
                                        --epochs 10 \
```

"""

print("-" * 10)
print("Building model")
print("-" * 10)
name_model = "HomographyModel_" + str(opt.lr) + "_" + str(opt.batch_size) + ".h5"
print("Saving the best model weights to {}".format(name_model))

full_model = DeepHomoModel()

if opt.weights is not None:
    full_model.load_weights(opt.weights)

model = full_model.model
preprocessing_fn = full_model.preprocessing

# define optomizer
optim = tf.keras.optimizers.Adam(opt.lr)
# define loss function
total_loss = "mse"

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        name_model, save_weights_only=True, save_best_only=True, mode="min"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=10, verbose=1, cooldown=10, min_lr=0.00000001
    ),
]

model.summary()

print("-" * 10)
print("Building dataset")
print("-" * 10)

x_train_dir = os.path.join(opt.data_dir, opt.x_train_dir)
homo_train_dir = os.path.join(opt.data_dir, opt.y_train_dir)
x_test_dir = os.path.join(opt.data_dir, opt.x_test_dir)
homo_test_dir = os.path.join(opt.data_dir, opt.y_test_dir)

full_dataset = HomographyDatasetBuilder(
    img_train_dir=x_train_dir,
    img_test_dir=x_test_dir,
    homo_train_dir=homo_train_dir,
    homo_test_dir=homo_test_dir,
    batch_size=opt.batch_size,
    preprocess_input=preprocessing_fn,
)

train_dataloader, valid_dataloader = full_dataset._get_dataloader()

print("-" * 10)
print("Launching the training")
print("-" * 10)

model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=opt.epochs,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)
