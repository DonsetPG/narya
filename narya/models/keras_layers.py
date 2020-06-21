from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import tensorflow as tf


def pyramid_layer(
    x, indx, activation="tanh", output_size=8, nb_neurons=[512, 512, 256, 128]
):
    """Fully connected layers to add at the end of a network.

    Arguments:
        x: a tf.keras Tensor as input
        indx: Integer, an index to add to the name of the layers
        activation: String, name of the activation function to add at the end
        output_size: Size of the last layer, number of outputs
        nb_neurons: Size of the Dense layer to add
    Returns:
        output: a tf.keras Tensor as output
    Raises:

    """
    dense_name_base = "full_" + str(indx)
    for indx, neuron in enumerate(nb_neurons):
        x = tf.keras.layers.Dense(
            neuron, name=dense_name_base + str(neuron) + "_" + str(indx)
        )(x)
    x = tf.keras.layers.Dense(output_size, name=dense_name_base + "output")(x)
    output = tf.keras.layers.Activation(activation)(x)
    return output
