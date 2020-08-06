# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def simple_swish(features):
    features = tf.convert_to_tensor(features)
    return features * tf.nn.sigmoid(features)


def hard_swish(features):
    features = tf.convert_to_tensor(features)
    return features * tf.nn.relu6(features + tf.constant(3.)) * (1. / 6.)


def identity(features):
    features = tf.convert_to_tensor(features)
    return tf.identity(features)
