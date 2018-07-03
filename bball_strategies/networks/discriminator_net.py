from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def leaky_relu(features, leaky_relu_alpha=0.2):
    return tf.maximum(features, leaky_relu_alpha * features)


def residual_block(name, inputs, n_filters, n_layers=2, residual_alpha=1.0, leaky_relu_alpha=0.2):
    """ Res Block
    Params
    ------
    name : string
        as res block name scope
    inputs : tensor
    n_filters : int
        number of filter in ConV
    n_layers : int
        number of layers in Res Block
    residual_alpha :
        output = residual * residual_alpha + inputs
    leaky_relu_alpha :
        output = tf.maximum(features, leaky_relu_alpha * features)

    Return
    ------
        residual * residual_alpha + inputs
    """
    with tf.variable_scope(name):
        next_input = inputs
        for i in range(n_layers):
            with tf.variable_scope('conv' + str(i)) as scope:
                normed = layers.layer_norm(next_input)
                nonlinear = leaky_relu(
                    normed, leaky_relu_alpha=leaky_relu_alpha)
                conv = tf.layers.conv1d(
                    inputs=nonlinear,
                    filters=n_filters,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation=None,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer()
                )
                next_input = conv
        return next_input * residual_alpha + inputs


def network(state, action, reuse=False, is_gail=True):
    """
    network structure is alike the value estimation in ppo
    state : shape=[batch_size, length, 11, 2]
    action : shape=[batch_size, length, 5, 2]

    """
    with tf.variable_scope('network', reuse=reuse):
        batch_size = tf.shape(state)[0]
        buffer_size = tf.shape(state)[1]
        input_ = tf.reshape(state, shape=[batch_size, buffer_size, functools.reduce(
            operator.mul, state.shape.as_list()[2:], 1)])
        if is_gail:
            action_ = tf.reshape(action, shape=[batch_size, 5*2])
            hyper_action = tf.layers.dense(
                inputs=action_,
                units=functools.reduce(operator.mul, state.shape.as_list()[2:], 1),
                activation=leaky_relu,
                kernel_initializer=layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer()
            )
            hyper_action = tf.reshape(hyper_action, shape=[
                                    batch_size, 1, functools.reduce(operator.mul, state.shape.as_list()[2:], 1)])
        else:
            action_ = tf.reshape(action, shape=[batch_size, buffer_size, 5*2])
            hyper_action = tf.layers.conv1d(
                inputs=action_,
                filters=functools.reduce(operator.mul, state.shape.as_list()[2:], 1),
                kernel_size = 5,
                strides = 1,
                padding = 'same',
                activation = leaky_relu,
                kernel_initializer = layers.xavier_initializer(),
                bias_initializer = tf.zeros_initializer()
            )
        sum_act_obs=tf.add(input_, hyper_action)
        conv_input=tf.layers.conv1d(
            inputs = sum_act_obs,
            filters = 128,
            kernel_size = 5,
            strides = 1,
            padding = 'same',
            activation = leaky_relu,
            kernel_initializer = layers.xavier_initializer(),
            bias_initializer = tf.zeros_initializer()
        )
        # residual block
        next_input=conv_input
        for i in range(4):
            res_block=residual_block(
                'Res' + str(i), next_input, n_filters = 128, n_layers = 2)
            next_input=res_block

        normed=layers.layer_norm(next_input)
        nonlinear=leaky_relu(normed)
        conv_output=tf.layers.conv1d(
            inputs=nonlinear,
            filters=1,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=leaky_relu,
            kernel_initializer=layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
        score_by_frame = tf.reshape(conv_output, [tf.shape(conv_output)[0], tf.shape(conv_output)[1]])
        scores = tf.reduce_mean(score_by_frame, axis=1)
        final_ = tf.reshape(
            scores, shape=[batch_size, ])
        def_value = tf.check_numerics(final_, 'def_value')

        return def_value, score_by_frame

        # init_xavier_weights = tf.variance_scaling_initializer(
        #     scale=1.0, mode='fan_avg', distribution='uniform')
        # init_output_weights = tf.variance_scaling_initializer(
        #     scale=0.1, mode='fan_in', distribution='normal')
        # conv1 = tf.layers.conv1d(
        #     inputs=input_,
        #     filters=128,
        #     kernel_size=3,
        #     padding='same',
        #     activation=tf.nn.relu,
        #     kernel_initializer=init_xavier_weights,
        # )
        # conv2 = tf.layers.conv1d(
        #     inputs=conv1,
        #     filters=128,
        #     kernel_size=3,
        #     padding='same',
        #     activation=tf.nn.relu,
        #     kernel_initializer=init_xavier_weights,
        # )
        # flatten = tf.reshape(conv2, shape=[batch_size, functools.reduce(
        #     operator.mul, conv2.shape.as_list()[1:], 1)])
        # # concat with action
        # action_ = tf.reshape(action, shape=[batch_size, 10])
        # concat_act_obs = tf.concat([flatten, action_], axis=1)
        # trunk_fc = tf.layers.dense(
        #     inputs=concat_act_obs,
        #     units=128,
        #     activation=tf.nn.relu,
        #     kernel_initializer=init_xavier_weights,
        # )
        # # defensive
        # def_fc = tf.layers.dense(
        #     inputs=trunk_fc,
        #     units=64,
        #     activation=tf.nn.relu,
        #     kernel_initializer=init_xavier_weights,
        # )
        # def_value = tf.layers.dense(
        #     inputs=def_fc,
        #     units=1,
        #     activation=None,
        #     kernel_initializer=init_output_weights,
        # )
        # def_value = tf.reshape(
        #     def_value, shape=[batch_size])
        # def_value = tf.check_numerics(def_value, 'def_value')

        # return def_value
