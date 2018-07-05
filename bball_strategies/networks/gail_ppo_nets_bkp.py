from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import agents
from .custom_distribution import CustomKLDiagNormal


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
    init_xavier_weights = tf.variance_scaling_initializer(
        scale=1.0, mode='fan_avg', distribution='uniform')
    with tf.variable_scope(name):
        next_input = inputs
        for i in range(n_layers):
            with tf.variable_scope('conv' + str(i)) as scope:
                # normed = layers.layer_norm(next_input)
                nonlinear = leaky_relu(
                    next_input, leaky_relu_alpha=leaky_relu_alpha)
                conv = tf.layers.conv2d(
                    inputs=nonlinear,
                    filters=n_filters,
                    kernel_size=[1, 5],
                    padding='same',
                    activation=None,
                    kernel_initializer=init_xavier_weights,
                    bias_initializer=tf.zeros_initializer()
                )
                next_input = conv
        return next_input * residual_alpha + inputs


def gail_def_gaussian(config, action_space, observations, unused_length, state=None):
    # observation space = shape=(batch_size, episode_length, 10, 14, 2)
    # action space = shape=(batch, episode_length, 23)
    with tf.variable_scope('defense'):
        batch_size = tf.shape(observations)[0]
        episode_len = tf.shape(observations)[1]
        input_ = tf.reshape(observations, shape=[batch_size, episode_len, observations.shape.as_list()[
                            2], functools.reduce(operator.mul, observations.shape.as_list()[3:], 1)])
        init_xavier_weights = tf.variance_scaling_initializer(
            scale=1.0, mode='fan_avg', distribution='uniform')
        init_output_weights = tf.variance_scaling_initializer(
            scale=config.init_output_factor, mode='fan_in', distribution='normal')

        with tf.variable_scope('policy'):
            conv_input = tf.layers.conv2d(
                inputs=input_,
                filters=128,
                kernel_size=[1, 5],
                padding='same',
                activation=None,
                kernel_initializer=init_xavier_weights,
                bias_initializer=tf.zeros_initializer()
            )
            # residual block
            next_input = conv_input
            for i in range(2):
                res_block = residual_block(
                    'Res' + str(i), next_input, n_filters=128, n_layers=2)
                next_input = res_block

            # normed = layers.layer_norm(next_input)
            nonlinear = leaky_relu(next_input)
            res_out = tf.layers.conv2d(
                inputs=nonlinear,
                filters=1,
                kernel_size=[1, 5],
                padding='same',
                activation=tf.tanh,
                kernel_initializer=init_output_weights,
                bias_initializer=tf.zeros_initializer()
            )
            def_action_mean = tf.reshape(res_out, shape=[batch_size, episode_len, 10])

        with tf.variable_scope('value'):
            conv_input = tf.layers.conv2d(
                inputs=input_,
                filters=128,
                kernel_size=[1, 5],
                padding='same',
                activation=None,
                kernel_initializer=init_xavier_weights,
                bias_initializer=tf.zeros_initializer()
            )
            # residual block
            next_input = conv_input
            for i in range(2):
                res_block = residual_block(
                    'Res' + str(i), next_input, n_filters=128, n_layers=2)
                next_input = res_block

            normed = layers.layer_norm(next_input)
            nonlinear = leaky_relu(normed)
            res_out = tf.layers.conv2d(
                inputs=nonlinear,
                filters=1,
                kernel_size=[1, 5],
                padding='same',
                activation=leaky_relu,
                kernel_initializer=init_xavier_weights,
                bias_initializer=tf.zeros_initializer()
            )
            flatten = tf.reshape(res_out, shape=[batch_size, episode_len, functools.reduce(
                operator.mul, res_out.shape.as_list()[2:], 1)])

            def_value = tf.layers.dense(
                inputs=flatten,
                units=1,
                activation=None,
                kernel_initializer=init_output_weights,
            )
            def_value = tf.reshape(
                def_value, shape=[batch_size, episode_len])
            def_value = tf.check_numerics(def_value, 'def_value')

        before_softplus_std_initializer = tf.constant_initializer(
            np.log(np.exp(config.init_std) - 1))
        def_actions_std = tf.nn.softplus(tf.get_variable(  # TODO
            'def_before_softplus_std', def_action_mean.shape[2:], tf.float32,
            before_softplus_std_initializer))
        def_actions_std = tf.tile(
            def_actions_std[None, None] + 1e-8,
            [tf.shape(observations)[0], tf.shape(observations)[1], 1])
        def_action_mean = tf.check_numerics(
            def_action_mean, 'def_action_mean')
        def_actions_std = tf.check_numerics(
            def_actions_std, 'def_actions_std')
        def_actions = CustomKLDiagNormal(
            def_action_mean, def_actions_std)
        def_policy = def_actions

    return agents.tools.AttrDict(state=state, policy=[def_policy], value=def_value)
