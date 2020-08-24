# -*- coding: utf - 8 -*-

import tensorflow as tf


# first Input
inputs = tf.keras.Input(shape=(32, 1), name='inputs')
h = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
h = tf.keras.layers.AveragePooling1D(3)(h)
h = tf.keras.layers.Flatten()(h)

# second Input
aux_inputs = tf.keras.Input(shape=(12,), name='aux_inputs')
h = tf.keras.layers.Concatenate()([h, aux_inputs])

# first output
outputs = tf.keras.layers.Dense(
    20, activation='sigmoid', name='outputs'
)(h)

# second output
aux_outputs = tf.keras.layers.Dense(
    1, activation='linear', name='aux_outputs'
)(h)

# Model
model = tf.keras.Model(
    inputs=[inputs, aux_inputs],
    outputs=[outputs, aux_outputs]
)

# compile
# 输出名字和 loss 名字一定要对应
model.compile(
    loss={
        'outputs': 'binary_crossentropy',
        'aux_outputs': 'mse'
    },
    loss_weights={
        'outputs': 1,
        'aux_outputs': 0.4
    },
    metrics=['accuracy']
)

# fit
# 输入名字和 Input 名字一定要对应
# 输出名字要和输出对应
his = model.fit(
    x={
        'inputs': X_train,
        'aux_inputs': X_aux
    },
    y={
        'outputs': y_train,
        'aux_outputs': y_aux
    },
    validation_split=0.2,
    epochs=20
)