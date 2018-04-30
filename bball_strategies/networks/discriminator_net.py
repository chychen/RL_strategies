from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def network(state, reuse=False):
    """
    network structure is alike the value estimation in ppo
    state : shape=[batch_size, buffer_size, 14, 2]

    """
    with tf.variable_scope('network', reuse=reuse):
        batch_size = tf.shape(state)[0]
        buffer_size = state.shape[1]
        input_ = tf.reshape(state, shape=[batch_size, buffer_size, functools.reduce(
            operator.mul, state.shape.as_list()[2:], 1)])
        init_xavier_weights = tf.variance_scaling_initializer(
            scale=1.0, mode='fan_avg', distribution='uniform')
        init_output_weights = tf.variance_scaling_initializer(
            scale=0.1, mode='fan_in', distribution='normal')
        conv1 = tf.layers.conv1d(
            inputs=input_,
            filters=128,
            kernel_size=3,
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        conv2 = tf.layers.conv1d(
            inputs=conv1,
            filters=128,
            kernel_size=3,
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        flatten = tf.reshape(conv2, shape=[batch_size, functools.reduce(
            operator.mul, conv2.shape.as_list()[1:], 1)])
        trunk_fc = tf.layers.dense(
            inputs=flatten,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        # defensive
        def_fc = tf.layers.dense(
            inputs=trunk_fc,
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        def_value = tf.layers.dense(
            inputs=def_fc,
            units=1,
            activation=None,
            kernel_initializer=init_output_weights,
        )
        def_value = tf.reshape(
            def_value, shape=[batch_size])
        def_value = tf.check_numerics(def_value, 'def_value')

        return def_value
