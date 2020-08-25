# -*- coding: utf - 8 -*-

import math
import tensorflow as tf
from layers.embedding_layers import word_embedding_layer


class Transformer(tf.keras.Model):

    def __init__(
            self,
            inputs_vocab_size,
            targets_vocab_size,
            embedding_size,

    ):
        pass

    def predict(self, encoder_outputs, encoder_decoder_attention_mask, training):
        pass

    def _get_decode_next_logits_fn(self, max_decode_len):
        """
            在函数内返回函数形成闭包，以保存 position_embeddings 和 look_ahead_mask
        """

        # 获取所有位置的位置编码
        # 在整个解码过程中会按照解码步骤的进行使用
        position_embeddings = self.position_embedding(
            inputs=None, length=max_decode_len + 1
        )
        position_embeddings = tf.cast(position_embeddings, self._dtype)

        # 获取 look ahead attention mask
        # 在整个解码过程中按照解码步骤的进行使用
        decode_look_ahead_mask = utils.get_look_ahead_mask(
            max_decode_len, dtype=self._dtype
        )

        def decode_next_logits(decoded_targets_ids, i, cache):
            """
            :param decoded_targets_ids: (batch_size * beam_size, i + 1)
            :param i: 已经解码 i 个 token
            :param cache: 缓存了各层的 attention
            :return:
            """

            # 取出上一个词，用来预测下一个词
            last_targets_id = decoded_targets_ids[:, -1]

            last_targets_embeddings = self.targets_embedding_lookup(
                last_targets_id
            )

            cross_attention_mask = cache.get('encoder_decoder_attention_mask')
            cross_attention_mask = tf.where(
                cross_attention_mask < 0,
                tf.zeros_like(cross_attention_mask),
                tf.ones_like(cross_attention_mask)
            )
            cross_attention_mask = tf.squeeze(cross_attention_mask, axis=1)
            cross_attention_mask = tf.tile(cross_attention_mask, [batch_size, 1, 1])

            decoder_outputs = self.decoder_layer(
                decoder_input,
                cross_attention_mask,

            )

