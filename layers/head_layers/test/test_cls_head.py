# -*- coding: utf - 8 -*-

import tensorflow as tf
from layers.head_layers import cls_head


class ClassificationHead(tf.test.TestCase):
    def test_layer_invocation(self):
        test_layer = cls_head.ClassificationHead(inner_dim=5, num_classes=2)
        features = tf.zeros(shape=(2, 10, 10), dtype=tf.float32)
        output = test_layer(features)
        self.assertAllClose(output, [[0, 0], [0, 0]])
        self.assertSameElements(
            test_layer.checkpoint_items.keys(), ['pooler_dense']
        )

    def test_layer_serialization(self):
        layer = cls_head.ClassificationHead(10, 2)
        new_layer = cls_head.ClassificationHead.from_config(layer.get_config())
        self.assertAllEqual(layer.get_config(), new_layer.get_config())


if __name__ == '__main__':
    tf.test.main()
