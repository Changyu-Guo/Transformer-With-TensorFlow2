# -*- coding: utf - 8 -*-

import tensorflow as tf
from ops import beam_search


class BeamSearchTest(tf.test.TestCase):

    def test_expand_to_beam_size(self):
        x = tf.ones([7, 4, 2, 5])
        x = beam_search._expand_to_beam_size(x, 3)
        shape = tf.shape(x)
        self.assertAllEqual([7, 3, 4, 2, 5], shape)

    def test_get_shape_keep_last_dim(self):
        y = tf.constant(4.0)
        x = tf.ones([7, tf.cast(tf.sqrt(y), tf.int32), 2, 5])
        shape = beam_search._get_shape_keep_last_dim(x)
        self.assertAllEqual([None, None, None, 5], shape.as_list())

    def test_flatten_beam_dim(self):
        x = tf.ones([7, 4, 2, 5])
        x = beam_search._flatten_beam_dim(x)
        self.assertAllEqual([28, 2, 5], tf.shape(x))

    def test_unflatten_beam_dim(self):
        x = tf.ones([28, 2, 5])
        x = beam_search._unflatten_beam_dim(x, 7, 4)
        self.assertAllEqual([7, 4, 2, 5], tf.shape(x))

    def test_gather_beams(self):
        x = tf.reshape(tf.range(24), [2, 3, 4])
        y = beam_search._gather_beams(x, [[1, 2], [0, 2]], 2, 2)
        self.assertAllEqual(
            [
                [[4, 5, 6, 7],
                [8, 9, 10, 11]],

                [[12, 13, 14, 15],
                 [20, 21, 22, 23]]
            ],
            y
        )

    def test_gather_topk_beams(self):
        x = tf.reshape(tf.range(24), [2, 3, 4])
        x_scores = [[0, 1, 1], [1, 0, 1]]

        y = beam_search._gather_topk_beams(x, x_scores, 2, 2)
        self.assertAllEqual(
            [
                [[4, 5, 6, 7],
                 [8, 9, 10, 11]],

                [[12, 13, 14, 15],
                 [20, 21, 22, 23]]
            ],
            y
        )


if __name__ == '__main__':
    tf.test.main()
