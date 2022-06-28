from tensorflow.keras.models import Model
from tensorflow.keras import layers, activations, regularizers, optimizers, initializers, constraints
from tensorflow.keras.regularizers import Regularizer

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np


class WeightMaskCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight_mask):
        super().__init__()
        self.weight_mask = weight_mask

    def on_train_batch_end(self, batch, logs=None):
        for i in range(len(self.model.layers)):
            new_weights = [self.model.layers[i].get_weights()[j] * self.weight_mask[i][j] for j in
                           range(len(self.model.layers[i].get_weights()))]
            self.model.layers[i].set_weights(new_weights)


class ClippedDense(tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        super(ClippedDense, self).__init__(**kwargs)
        self.weight_thresh = None

    def set_weight_tresh(self, weight_tresh):
        self.weight_thresh = weight_tresh

    def call(self, inputs):
        if self.weight_thresh:
            kernel_mask = tf.cast(tf.abs(self.kernel) > self.weight_tresh, dtype=tf.float32)
            self.kernel = tf.multiply(self.kernel * kernel_mask)

            bias_mask = tf.cast(tf.abs(self.bias) > self.weight_tresh, dtype=tf.float32)
            self.bias = tf.multiply(self.bias * bias_mask)
        return tf.matmul(inputs, self.kernel) + self.bias


class L1L2_m(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.01):
        with K.name_scope(self.__class__.__name__):
            self.l1 = K.variable(l1, name='l1')
            self.l2 = K.variable(l2, name='l2')
            self.val_l1 = l1
            self.val_l2 = l2

    def set_l1_l2(self, l1, l2):
        K.set_value(self.l1, l1)
        K.set_value(self.l2, l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def __call__(self, x):
        regularization = 0.
        if self.val_l1 > 0.:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.val_l2 > 0.:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        config = {'l1': float(K.get_value(self.l1)),
                  'l2': float(K.get_value(self.l2))}
        return config


def set_model_l1_l2(model, l1, l2):
    for layer in model.layers:
        if 'kernel_regularizer' in dir(layer) and \
                isinstance(layer.kernel_regularizer, L1L2_m):
            layer.kernel_regularizer.set_l1_l2(l1, l2)


def log_activation(in_x):
    return tf.math.log(in_x)


def abs_activation(in_x):
    return tf.math.abs(in_x)


def sign_activation(in_x):
    return tf.math.sign(in_x)


def eql_model(input_size):
    inputs_x = [layers.Input(shape=(1,)) for i in range(input_size)]
    ln_layers = [layers.Dense(1, use_bias=False,
                              kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                              kernel_initializer=initializers.Identity(gain=1.0),
                              trainable=False,
                              activation=log_activation)(input_x) for input_x in inputs_x]
    ln_concat = layers.Concatenate()(ln_layers)
    ln_dense = layers.Dense(1,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                            use_bias=False, activation=activations.exponential,
                            kernel_initializer=initializers.Identity(gain=1.0))(ln_concat)

    input_x_concat = layers.Concatenate()(inputs_x)
    input_x_dense = layers.Dense(1,
                                 activation='linear',
                                 kernel_regularizer=regularizers.l1_l2(
                                     l1=1e-3,
                                     l2=1e-3),
                                 bias_regularizer=regularizers.l1_l2(
                                     l1=1e-3,
                                     l2=1e-3))(input_x_concat)

    output_concat = layers.Concatenate()([ln_dense, input_x_dense])
    output_dense = layers.Dense(1, use_bias=False, activation='linear',
                                kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))(output_concat)
    model = Model(inputs=inputs_x, outputs=output_dense, name='eql_model')

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.01,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)
    opt = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


def eql_ln_block(inputs_x, layer_num):
    ln_layers = [layers.Dense(1, use_bias=False,
                              kernel_initializer=initializers.Identity(gain=1.0),
                              trainable=False,
                              activation=log_activation,
                              name='ln_{}_{}'.format(layer_num, i))(input_x) for i, input_x in enumerate(inputs_x)]
    ln_concat = layers.Concatenate()(ln_layers)
    ln_dense = layers.Dense(1,
                            kernel_regularizer=L1L2_m(l1=1e-3, l2=1e-3),
                            use_bias=False, activation=activations.exponential,
                            kernel_initializer=initializers.Identity(gain=1.0),
                            name='ln_dense_{}'.format(layer_num))(ln_concat)
    return ln_dense


def eql_model_v2(input_size, ln_block_count=2, decay_steps=1000, linear_block=False,
                 compile=True):
    inputs_x = [layers.Input(shape=(1,)) for i in range(input_size)]
    ln_dense_units = [eql_ln_block(inputs_x, layer_num=i) for i in range(ln_block_count)]

    if ln_block_count == 1:
        if linear_block:
            ln_dense_concat = layers.Concatenate()(inputs_x + ln_dense_units)
        else:
            ln_dense_concat = ln_dense_units[0]
    else:
        if linear_block:
            ln_dense_concat = layers.Concatenate()(inputs_x + ln_dense_units)
        else:
            ln_dense_concat = layers.Concatenate()(ln_dense_units)
    output_dense = layers.Dense(1, activation='linear',
                                kernel_regularizer=L1L2_m(l1=1e-3, l2=1e-3),
                                bias_regularizer=L1L2_m(l1=1e-3, l2=1e-3),
                                name='output_dense')(ln_dense_concat)
    model = Model(inputs=inputs_x, outputs=output_dense, name='eql_model')

    if compile:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.01,
            decay_steps=decay_steps,
            decay_rate=0.96,
            staircase=True)
        opt = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


def add_ln_block(input_size, trained_model, cur_ln_block_count, decay_steps=1000, freeze_prev=False):
    model = eql_model_v2(input_size, ln_block_count=cur_ln_block_count + 1, decay_steps=decay_steps,
                         compile=False)
    for i in range(cur_ln_block_count):
        ln_block = 'ln_dense_{}'.format(i)
        model.get_layer(ln_block).set_weights(trained_model.get_layer(ln_block).get_weights())
        if freeze_prev:
            model.get_layer(ln_block).trainable = False
    trained_output_kernel = trained_model.get_layer('output_dense').get_weights()[0]
    trained_output_bias = trained_model.get_layer('output_dense').get_weights()[1]
    init_output_kernel = model.get_layer('output_dense').get_weights()[0]
    init_output_bias = model.get_layer('output_dense').get_weights()[1]
    print(init_output_kernel)
    print(init_output_bias)
    model.get_layer('output_dense').set_weights([np.append(trained_output_kernel, [init_output_kernel[-1]], axis=0),
                                                 init_output_bias])
    return model


def eql_model_signed(input_size):
    inputs_x = [layers.Input(shape=(1,)) for i in range(input_size)]

    abs_layers = [layers.Dense(1, use_bias=False,
                               kernel_initializer=initializers.Identity(gain=1.0),
                               trainable=False,
                               activation=abs_activation,
                               name='in_abs_{}'.format(i))(input_x) for i, input_x in enumerate(inputs_x)]
    ln_layers = [layers.Dense(1, use_bias=False,
                              kernel_initializer=initializers.Identity(gain=1.0),
                              trainable=False,
                              activation=log_activation,
                              name='in_log_{}'.format(i))(input_x) for i, input_x in enumerate(abs_layers)]
    ln_concat = layers.Concatenate()(ln_layers)

    ln_dense = layers.Dense(1,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                            use_bias=False, activation=activations.exponential,
                            kernel_initializer=initializers.Identity(gain=1.0),
                            name='log_dense')(ln_concat)

    sign_layers = [layers.Dense(1, use_bias=False,
                                kernel_initializer=initializers.Identity(gain=1.0),
                                trainable=False,
                                activation=sign_activation,
                                name='sign_{}'.format(i))(input_x) for i, input_x in enumerate(inputs_x)]
    sign_concat = layers.Multiply()(sign_layers)

    ln_dense_with_sign = layers.Concatenate()([ln_dense, sign_concat])
    ln_dense_signed = layers.Dense(1, use_bias=False,
                                   kernel_initializer=initializers.Identity(gain=1.0),
                                   trainable=False,
                                   name='log_sign')(ln_dense_with_sign)
    input_x_concat = layers.Concatenate()(inputs_x)
    input_x_dense = layers.Dense(1,
                                 activation='linear',
                                 kernel_regularizer=regularizers.l1_l2(
                                     l1=1e-3,
                                     l2=1e-3),
                                 bias_regularizer=regularizers.l1_l2(
                                     l1=1e-3,
                                     l2=1e-3),
                                 name='input_dense')(input_x_concat)

    output_concat = layers.Concatenate()([ln_dense_signed, input_x_dense])
    output_dense = layers.Dense(1, use_bias=False, activation='linear',
                                kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3),
                                name='output_dense')(output_concat)
    model = Model(inputs=inputs_x, outputs=output_dense, name='eql_model')

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.01,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True)
    opt = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


if __name__ == "__main__":
    model = eql_model_v2(2, ln_block_count=1)
    model_2 = add_ln_block(2, model, 1)
    print(model_2.summary())
    print(model_2.get_weights())
    # print(model.optimizer.get_gradients(model.total_loss, model.trainable_weights))
    # print(model.summary())
