import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D


def squash(s, axis=-2, epsilon=1e-8, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm

        return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=keepdims)
        return tf.sqrt(squared_norm + epsilon)


def loss_margin(logits, labels, n_classes, clip_rate=0.1, lam=0.5):
    m_plus, m_minus = 1 - clip_rate, clip_rate

    with tf.variable_scope('loss_margin'):
        present_error = tf.square(tf.maximum(0., m_plus - logits), name="present_error")
        absent_error = tf.square(tf.maximum(0., logits - m_minus), name="absent_error")

        T = tf.one_hot(labels, depth=n_classes, name="T")
        L = T * present_error + lam * (1.0 - T) * absent_error
        loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="loss_margin")

        return loss


class PrimaryCaps(tf.keras.layers.Layer):
    def __init__(self, caps1_n_caps, caps1_n_dims, fast=True):
        super(PrimaryCaps, self).__init__()

        self.caps1_n_caps, self.caps1_n_dims = caps1_n_caps, caps1_n_dims

        # Define this bottleneck from outside (maybe)
        # caps1_n_maps = 32
        #
        # act = tf.nn.relu
        # conv1_params = {
        #     "filters": 256,
        #     "kernel_size": 9,
        #     "strides": 1,
        #     "padding": "valid",
        #     "activation": act
        # }
        # conv2_params = {
        #     "filters": caps1_n_maps * caps1_n_dims,
        #     "kernel_size": 9,
        #     "strides": 2,
        #     "padding": "valid",
        #     "activation": act
        # }
        #
        # self.conv1 = Conv2D(**conv1_params)
        # self.conv2 = Conv2D(**conv2_params)

    def call(self, inputs, *args, **kwargs):
        # Bottleneck
        # out = self.conv2d(inputs)  # [batch_size, 20, 20, 256]
        # out = self.conv2d(out)  # [batch_size, caps1_n_caps, caps1_n_dims, 256]
        out = inputs

        # Create primary capsules
        capsules = tf.reshape(out, [-1, self.caps1_n_caps, 1, self.caps1_n_dims, 1])
        primary_caps = squash(capsules)  # [batch_size, caps1_n_caps, 1, caps1_n_dims, 1]

        return primary_caps


class DigitalCaps(tf.keras.layers.Layer):
    def __init__(self, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims, fast=True):
        super(DigitalCaps, self).__init__()

        self.fast = fast
        self.caps2_n_caps, self.caps2_n_dims = caps2_n_caps, caps2_n_dims

        if self.fast:
            self.w = self.add_weight('W', shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        else:
            self.w = self.add_weight('W', shape=[1, caps1_n_caps, caps2_n_dims * caps2_n_caps, caps1_n_dims, 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))

    def call(self, primary_caps, *args, **kwargs):

        if self.fast:
            w_tiled = tf.tile(self.w, [tf.shape(primary_caps)[0], 1, 1, 1, 1], name="W_tiled")

            primary_caps_tiled = tf.tile(primary_caps, [1, 1, self.caps2_n_caps, 1, 1],
                                         name="primary_caps_tiled")

            digit_caps = tf.matmul(w_tiled, primary_caps_tiled, name="digit_caps")
        else:
            primary_caps_tiled = tf.tile(primary_caps, [1, 1, self.caps2_n_dims * self.caps2_n_caps, 1, 1])
            digit_caps = tf.reduce_sum(self.w * primary_caps_tiled, axis=3, keepdims=True)
            digit_caps = tf.reshape(digit_caps,
                                    shape=[-1, primary_caps.shape[1], self.caps2_n_caps, self.caps2_n_dims, 1])

        digit_caps_stopped = tf.stop_gradient(digit_caps, name='stop_gradient')

        return digit_caps, digit_caps_stopped


class DynamicRouting(tf.keras.layers.Layer):
    def __init__(self, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims):
        super(DynamicRouting, self).__init__()

        self.caps1_n_caps = caps1_n_caps
        self.caps1_n_dims = caps1_n_dims

        self.caps1_n_caps, self.caps1_n_dims, self.caps2_n_caps, self.caps2_n_dims = \
            caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims

        self.bias = self.add_weight('B', shape=(1, 1, caps2_n_caps, caps2_n_dims, 1))

    def call(self, digit_caps, digit_caps_stopped, routing_iter, *args, **kwargs):
        digit_caps_round = None
        self.routing_iter = routing_iter

        raw_weights = tf.zeros([tf.shape(digit_caps)[0], self.caps1_n_caps, self.caps2_n_caps, 1, 1], dtype=tf.float32,
                               name="raw_weights")
        for r_iter in range(self.routing_iter):
            with tf.variable_scope('routing_round_{}'.format(r_iter)):
                digit_caps_iter = digit_caps if r_iter == routing_iter - 1 else digit_caps_stopped

                routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")

                weighted_predictions = tf.multiply(routing_weights, digit_caps_iter,
                                                   name="weighted_predictions")
                weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True,
                                             name="weighted_sum") + self.bias
                digit_caps_round = squash(weighted_sum, axis=-2, name="digit_caps_round")

                if r_iter != self.routing_iter - 1:
                    digit_caps_tiled = tf.tile(digit_caps_round, [1, self.caps1_n_caps, 1, 1, 1],
                                               name="digit_caps_tiled")
                    agreement = tf.reduce_sum(digit_caps_iter * digit_caps_tiled, axis=3, keepdims=True)
                    raw_weights += agreement

        return digit_caps_round


class Capsule(tf.keras.layers.Layer):
    def __init__(self, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims, routing_iters=3):
        super(Capsule, self).__init__()

        # Settings
        self.routing_iters = routing_iters
        shapes = (caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims)

        # Layers
        self.primary_caps = PrimaryCaps(caps1_n_caps, caps1_n_dims)
        self.digital_caps = DigitalCaps(*shapes, fast=True)
        self.dynamic_routing = DynamicRouting(*shapes)

    def call(self, inputs, *args, **kwargs):

        primary_caps = self.primary_caps(inputs)
        digit_caps, digit_caps_stopped = self.digital_caps(primary_caps)
        digit_caps_round = self.dynamic_routing(digit_caps, digit_caps_stopped, self.routing_iters)

        return digit_caps_round
