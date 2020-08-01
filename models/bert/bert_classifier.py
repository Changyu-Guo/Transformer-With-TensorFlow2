# -*- coding: utf - 8 -*-

import tensorflow as tf
from networks import regression_and_classification


class BertClassifier(tf.keras.Model):
    def __init__(
            self,
            network,
            num_classes,
            initializer='glorot_uniform',
            dropout_rate=0.1,
            use_encoder_pooler=True,
            **kwargs
    ):
        inputs = network.inputs

        if use_encoder_pooler:
            _, cls_output = network(inputs)
            cls_output = tf.keras.layers.Dropout(rate=dropout_rate)(cls_output)
            self.classifier = regression_and_classification.Classification(
                input_size=cls_output.shape[-1],
                num_classes=num_classes,
                initializer=initializer,
                output='logits',
                name='sentence_prediction'
            )
            predictions = self.classifier(cls_output)
        else:
            sequence_output, _ = network(inputs)
            self.classifier = None
            predictions = self.classifier(sequence_output)

        super(BertClassifier, self).__init__(
            inputs=inputs, outputs=predictions, **kwargs
        )