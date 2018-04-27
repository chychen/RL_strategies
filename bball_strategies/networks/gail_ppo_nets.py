from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import agents
from custom_distribution import CustomKLDiagNormal


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
            conv1 = tf.layers.conv2d(
                inputs=input_,
                filters=128,
                kernel_size=[1, 3],
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=init_xavier_weights,
            )
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=128,
                kernel_size=[1, 3],
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=init_xavier_weights,
            )
            flatten = tf.reshape(conv2, shape=[batch_size, episode_len, functools.reduce(
                operator.mul, conv2.shape.as_list()[2:], 1)])
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
            def_action_mean = tf.layers.dense(
                inputs=def_fc,
                units=10,
                activation=tf.tanh,
                kernel_initializer=init_output_weights,
            )

        with tf.variable_scope('value'):
            conv1 = tf.layers.conv2d(
                inputs=input_,
                filters=128,
                kernel_size=[1, 3],
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=init_xavier_weights,
            )
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=128,
                kernel_size=[1, 3],
                padding='valid',
                activation=tf.nn.relu,
                kernel_initializer=init_xavier_weights,
            )
            flatten = tf.reshape(conv2, shape=[batch_size, episode_len, functools.reduce(
                operator.mul, conv2.shape.as_list()[2:], 1)])
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
                def_value, shape=[batch_size, episode_len])
            def_value = tf.check_numerics(def_value, 'def_value')

        before_softplus_std_initializer = tf.constant_initializer(
            np.log(np.exp(config.init_std) - 1))
        def_actions_std = tf.nn.softplus(tf.get_variable(  # TODO
            'def_before_softplus_std', def_action_mean.shape[2:], tf.float32,
            before_softplus_std_initializer))
        def_actions_std = tf.tile(
            def_actions_std[None, None],
            [tf.shape(observations)[0], tf.shape(observations)[1], 1])
        def_action_mean = tf.check_numerics(
            def_action_mean, 'def_action_mean')
        def_actions_std = tf.check_numerics(
            def_actions_std, 'def_actions_std')
        def_actions = CustomKLDiagNormal(
            def_action_mean, def_actions_std)
        def_policy = def_actions

    return agents.tools.AttrDict(state=state, policy=[def_policy], value=def_value)
