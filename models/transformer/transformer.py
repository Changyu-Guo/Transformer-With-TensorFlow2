# -*- coding: utf - 8 -*-

import math
import tensorflow as tf
from layers.embedding_layers import word_embedding_layer
from layers.embedding_layers import transformer_position_embedding_layer
from layers.transformer_layers.encoder_stack import TransformerEncoderStack
from layers.transformer_layers.decoder_stack import TransformerDecoderStack
from layers import utils
from ops import beam_search
from metrics import transformer_metrics

BOS_ID = 0
EOS_ID = 1


def create_model(params, is_train):
    encoder_decoder_kwargs = dict(
        num_hidden_layers=params['num_hidden_layers'],
        num_attention_heads=params['num_attention_heads'],
        intermediate_size=params['intermediate_size'],
        intermediate_activation='relu',
        hidden_dropout_rate=params['hidden_dropout_rate'],
        attention_dropout_rate=params['attention_dropout_rate'],
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6
    )
    encoder_stack = TransformerEncoderStack(**encoder_decoder_kwargs)
    decoder_stack = TransformerDecoderStack(**encoder_decoder_kwargs)

    model_kwargs = dict(
        inputs_vocab_size=params['inputs_vocab_size'],
        targets_vocab_size=params['targets_vocab_size'],
        hidden_size=params['hidden_size'],
        attention_dropout_rate=params['attention_dropout_rate'],
        hidden_dropout_rate=params['hidden_dropout_rate'],
        max_decode_len=params['max_decode_len'],
        extra_decode_len=params['extra_decode_len'],
        beam_size=params['beam_size'],
        alpha=params['alpha'],
        encoder_stack=encoder_stack,
        decoder_stack=decoder_stack,
        dtype=params['dtype'],
        name='transformer'
    )

    if is_train:
        inputs_ids = tf.keras.layers.Input((None,), dtype=tf.int64, name='inputs_ids')
        targets_ids = tf.keras.layers.Input((None,), dtype=tf.int64, name='targets_ids')
        internal_model = Transformer(**model_kwargs)
        logits = internal_model([inputs_ids, targets_ids], training=is_train)
        targets_vocab_size = params['targets_vocab_size']
        label_smoothing = params['label_smoothing']
        if params['enable_metrics_in_training']:
            transformer_metrics.MetricLayer(targets_vocab_size)([logits, targets_ids])
        logits = tf.keras.layers.Lambda(lambda x: x, name='logits', dtype=tf.float32)(logits)
        model = tf.keras.Model([inputs_ids, targets_ids], logits)
        loss = transformer_metrics.transformer_loss(
            logits,
            targets_ids,
            label_smoothing,
            targets_vocab_size
        )
        model.add_loss(loss)
        return model

    else:
        batch_size = None
        inputs_ids = tf.keras.layers.Input(
            (None,),
            batch_size=batch_size,
            dtype=tf.int64,
            name='inputs_ids'
        )
        internal_model = Transformer(**model_kwargs)
        ret = internal_model([inputs_ids], training=is_train)
        outputs, scores = ret['outputs'], ret['scores']
        return tf.keras.Model(inputs_ids, [outputs, scores])


class Transformer(tf.keras.Model):

    def __init__(
            self,
            inputs_vocab_size,
            targets_vocab_size,
            hidden_size,
            attention_dropout_rate,
            hidden_dropout_rate,
            max_decode_len,
            extra_decode_len,
            beam_size,
            alpha,
            encoder_stack,
            decoder_stack,
            dtype=tf.float32,
            **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)

        self._inputs_vocab_size = inputs_vocab_size
        self._targets_vocab_size = targets_vocab_size
        self._hidden_size = hidden_size
        self._attention_dropout_rate = attention_dropout_rate
        self._hidden_dropout_rate = hidden_dropout_rate
        self._max_decode_len = max_decode_len
        self._extra_decode_len = extra_decode_len
        self._beam_size = beam_size
        self._alpha = alpha
        self._dtype = dtype

        # word embedding
        self.inputs_word_embedding = word_embedding_layer.WordEmbedding(
            vocab_size=self._inputs_vocab_size,
            embedding_size=self._hidden_size,
            initializer=tf.random_normal_initializer(
                mean=0, stddev=self._hidden_size ** -0.5
            ),
            use_scale=True
        )
        self.targets_word_embedding = word_embedding_layer.WordEmbedding(
            vocab_size=self._targets_vocab_size,
            embedding_size=self._hidden_size,
            initializer=tf.random_normal_initializer(
                mean=0, stddev=self._hidden_size ** -0.5
            ),
            use_scale=True
        )

        # positional encoding
        self.position_embedding = transformer_position_embedding_layer.TransformerPositionEmbedding(
            hidden_size=self._hidden_size
        )

        # encoder and decoder
        self.encoder_stack = encoder_stack
        self.decoder_stack = decoder_stack

        self.pre_encoder_dropout = tf.keras.layers.Dropout(rate=self._hidden_dropout_rate)
        self.pre_decoder_dropout = tf.keras.layers.Dropout(rate=self._hidden_dropout_rate)

    def call(self, inputs, training):
        if len(inputs) == 2:
            inputs_ids, targets_ids = inputs[0], inputs[1]
        else:
            inputs_ids, targets_ids = inputs[0], None

        with tf.name_scope('transformer'):

            # (batch_size, 1, seq_len)
            padding_mask = utils.get_padding_mask(
                inputs_ids,
                padding_value=0,
                dtype=self._dtype
            )

            encoder_outputs = self.encode(inputs_ids, padding_mask, training)

            if targets_ids is None:
                return self.predict(encoder_outputs, padding_mask, training)
            else:
                decoder_outputs = self.decode(targets_ids, encoder_outputs, padding_mask, training)
                logits = self.targets_word_embedding(decoder_outputs, mode='linear')
                logits = tf.cast(logits, tf.float32)
                return logits

    def encode(self, inputs_ids, padding_mask, training):

        with tf.name_scope('encode'):

            embedded_inputs = self.inputs_word_embedding(
                inputs_ids, mode='embedding'
            )
            embedded_inputs = tf.cast(embedded_inputs, self._dtype)

            position_embedding = self.position_embedding(embedded_inputs)

            encoder_inputs = embedded_inputs + position_embedding

            if training:
                encoder_inputs = self.pre_encoder_dropout(encoder_inputs)

            encoder_outputs = self.encoder_stack(encoder_inputs, padding_mask, training)

            return encoder_outputs

    def decode(self, targets_ids, encoder_outputs, padding_mask, training):

        with tf.name_scope('decode'):

            embedded_targets = self.targets_word_embedding(targets_ids)
            embedded_targets = tf.cast(embedded_targets, self._dtype)
            # 假定句子中已经有 BOS 和 EOS
            embedded_targets = embedded_targets[:, :-1, :]

            position_embedding = self.position_embedding(embedded_targets)
            position_embedding = tf.cast(position_embedding, self._dtype)

            decoder_inputs = embedded_targets + position_embedding

            if training:
                decoder_inputs = self.pre_decoder_dropout(decoder_inputs)

            targets_seq_len = tf.shape(targets_ids)[1]
            # (1, seq_len, seq_len)
            look_ahead_mask = utils.get_look_ahead_mask(targets_seq_len)

            decoder_outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                padding_mask,
                look_ahead_mask,
                training
            )

            return decoder_outputs

    def predict(self, encoder_outputs, padding_mask, training):
        max_decode_len = self._max_decode_len or (
                tf.shape(encoder_outputs)[1] + self._extra_decode_len
        )

        decode_next_logits_fn = self._get_decode_next_logits_fn(max_decode_len, training=training)

        batch_size = tf.shape(encoder_outputs)[0]
        initial_ids = tf.fill((batch_size,), BOS_ID)
        initial_ids = tf.cast(initial_ids, tf.int32)

        init_decode_length = 0
        num_heads = self.decoder_stack.num_attention_heads
        size_per_head = self._hidden_size // num_heads

        cache = {
            str(layer): {
                'key':
                    tf.zeros(
                        shape=(batch_size, init_decode_length, num_heads, size_per_head),
                        dtype=self._dtype
                    ),
                'value':
                    tf.zeros(
                        shape=(batch_size, init_decode_length, num_heads, size_per_head),
                        dtype=self._dtype
                    )
            } for layer in range(self.decoder_stack.num_hidden_layers)
        }

        cache['encoder_outputs'] = encoder_outputs
        cache['padding_mask'] = padding_mask

        decoded_ids, scores = beam_search.sequence_beam_search(
            decode_next_logits_fn=decode_next_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self._targets_vocab_size,
            beam_size=self._beam_size,
            alpha=self._alpha,
            max_decode_length=max_decode_len,
            eos_id=EOS_ID,
            dtype=self._dtype
        )

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {
            'outputs': top_decoded_ids,
            'scores': top_scores
        }

    def _get_decode_next_logits_fn(self, max_decode_len, training):
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
        # (1, seq_len, seq_len)
        look_ahead_mask = utils.get_look_ahead_mask(
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
            # (batch_size * beam_size, 1)
            last_targets_ids = decoded_targets_ids[:, -1:]
            last_targets_embeddings = self.targets_word_embedding(
                last_targets_ids,
                mode='embedding'
            )
            last_targets_embeddings += position_embeddings[i: i+1]
            last_targets_mask = look_ahead_mask[:, i: i + 1, :i + 1]

            padding_mask = cache.get('padding_mask')

            decoder_outputs = self.decoder_stack(
                last_targets_embeddings,
                cache.get('encoder_outputs'),
                padding_mask,
                last_targets_mask,
                training=training,
                cache=cache
            )

            logits = self.targets_word_embedding(
                decoder_outputs,
                mode='linear'
            )
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return decode_next_logits

    def get_config(self):
        config = {
            'inputs_vocab_size': self._inputs_vocab_size,
            'targets_vocab_size': self._targets_vocab_size,
            'hidden_size': self._hidden_size,
            'attention_dropout_rate': self._attention_dropout_rate,
            'hidden_dropout_rate': self._hidden_dropout_rate,
            'max_decode_len': self._max_decode_len,
            'extra_decode_len': self._extra_decode_len,
            'beam_size': self._beam_size,
            'encoder_stack': self.encoder_stack,
            'decoder_stack': self.decoder_stack,
            'dtype': self._dtype
        }
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
