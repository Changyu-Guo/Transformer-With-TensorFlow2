# -*- coding: utf - 8 -*-

import os
import tempfile
import tensorflow as tf
import numpy as np
from absl import logging
logging.set_verbosity(logging.INFO)

tempdir = tempfile.mkdtemp()

file = tf.keras.utils.get_file()
