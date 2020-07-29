# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, hidden_size, warmup_steps):
        super(LearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.hidden_size = hidden_size
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, global_step):
        with tf.name_scope('learning_rate_schedule'):
            global_step = tf.cast(global_step, tf.float32)
            learning_rate = self.initial_learning_rate
            # from Attention is all you need
            learning_rate *= tf.minimum(1.0, global_step / self.warmup_steps)
            learning_rate /= tf.sqrt(tf.maximum(global_step, self.warmup_steps))
            return learning_rate

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'hidden_size': self.hidden_size,
            'warmup_steps': self.warmup_steps
        }
