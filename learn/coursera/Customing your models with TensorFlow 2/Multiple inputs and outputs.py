# -*- coding: utf - 8 -*-

import tensorflow as tf
from tensorflow.keras import layers


inputs = tf.keras.Input(shape=(32, 1))
h = layers.Conv1D(16, 5, activation='relu')(inputs)
h = layers.AveragePooling1D(3)(h)
h = layers.Flatten()(h)

aux_inputs = tf.keras.Input(shape=(12,))
h = layers.Concatenate()([h, aux_inputs])
outputs = layers.Dense(20, activation='sigmoid')(h)
aux_outputs = layers.Dense(1, activation='linear')(h)

model = tf.keras.Model(
    inputs=[inputs, aux_inputs],
    outputs=[outputs, aux_outputs]
)

model.compile(
    loss=['binary_crossentropy', 'mse'],
    loss_weights=[1, 1],
    metrics=['accuracy']
)

his = model.fit(
    x=[X_train, X_aux],
    y=[y_train, y_aux],
    validation_split=0.2,
    epochs=20
)