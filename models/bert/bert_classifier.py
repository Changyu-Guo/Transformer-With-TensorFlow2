# -*- coding: utf - 8 -*-

import tensorflow as tf
from layers.head_layers.cls_head import ClassificationHead
from networks.classification import Classification


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
        self._self_setattr_tracking = False
        self._network = network
        self._config = {
            'network': network,
            'num_classes': num_classes,
            'initializer': initializer,
            'use_encoder_pooler': use_encoder_pooler
        }

        # bert_encoder
        inputs = network.inputs

        if use_encoder_pooler:
            _, cls_output = network(inputs)
            cls_output = tf.keras.layers.Dropout(rate=dropout_rate)(cls_output)
            self.classifier = classification.Classification(
                input_size=cls_output.shape[-1],
                num_classes=num_classes,
                initializer=initializer,
                output='logits',
                name='sentence_prediction'
            )
            predictions = self.classifier(cls_output)
        else:
            sequence_output, _ = network(inputs)
            self.classifier = ClassificationHead(
                inner_dim=sequence_output.shape[-1],
                num_classes=num_classes,
                initializer=initializer,
                dropout_rate=dropout_rate,
                name='sentence_prediction'
            )
            predictions = self.classifier(sequence_output)

        super(BertClassifier, self).__init__(
            inputs=inputs, outputs=predictions, **kwargs
        )

    @property
    def checkpoint_items(self):
        return dict(encoder=self._network)

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
