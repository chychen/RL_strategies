# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy networks for agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import agents

tfd = tf.contrib.distributions


# TensorFlow's default implementation of the KL divergence between two
# tf.contrib.distributions.MultivariateNormalDiag instances sometimes results
# in NaN values in the gradients (not in the forward pass). Until the default
# implementation is fixed, we use our own KL implementation.
class CustomKLDiagNormal(tfd.MultivariateNormalDiag):
    """Multivariate Normal with diagonal covariance and our custom KL code."""
    pass


@tfd.RegisterKL(CustomKLDiagNormal, CustomKLDiagNormal)
def _custom_diag_normal_kl(lhs, rhs, name=None):  # pylint: disable=unused-argument
    """Empirical KL divergence of two normals with diagonal covariance.

    Args:
      lhs: Diagonal Normal distribution.
      rhs: Diagonal Normal distribution.
      name: Name scope for the op.

    Returns:
      KL divergence from lhs to rhs.
    """
    with tf.name_scope(name or 'kl_divergence'):
        mean0 = lhs.mean()
        mean1 = rhs.mean()
        logstd0 = tf.log(lhs.stddev())
        logstd1 = tf.log(rhs.stddev())
        logstd0_2, logstd1_2 = 2 * logstd0, 2 * logstd1
        return 0.5 * (
            tf.reduce_sum(tf.exp(logstd0_2 - logstd1_2), -1) +
            tf.reduce_sum((mean1 - mean0) ** 2 / tf.exp(logstd1_2), -1) +
            tf.reduce_sum(logstd1_2, -1) - tf.reduce_sum(logstd0_2, -1) -
            mean0.shape[-1].value)


def two_trunk_gaussian(config, action_space, observations, unused_length, state=None):
    """
    ### Structure
    ### O_Trunk : offensive crown, shape=(15)
        - policy : [Categorical(3), CustomKLDiagNormal(11)]
            3 for discrete decisions, 1 for ball's direction, 10 for five ofensive players' dash(power, direction).
        - value : shape=(1)

    ### D_Trunk : defensive crown, shape=(11)
        - policy : [CustomKLDiagNormal(10)]
            10 for five defensive players' dash(power, direction)
        - value : shape=(1)

    Args
    ----
    config : Configuration object.
    action_space : Action space of the environment.
    observations : shape=[batch_size, episode_length, 5, 14, 2]
        Sequences of observations.
    unused_length : Batch of sequence lengths.
    state : Unused batch of initial states.

    Raises:
        ValueError: Unexpected action space.

    Returns
    -------
    Attribute dictionary containing the policy, value, and unused state.
    - policy : [Categorical(3), CustomKLDiagNormal(11), CustomKLDiagNormal(10)]
    - value : [off_value, def_value]

    TODO
    maybe softmax will limit the exploration ability
    tf.contrib.distributions.TransformedDistribution 或許可考慮？！
    because the action space might? like lognormal? than gaussian
    """
    # observation space = shape=(batch_size, episode_length, 5, 14, 2)
    # action space = shape=(batch, episode_length, 11, 2)
    batch_size = tf.shape(observations)[0]
    episode_len = tf.shape(observations)[1]

    # config
    init_xavier_weights = tf.variance_scaling_initializer(
        scale=1.0, mode='fan_avg', distribution='uniform')
    init_output_weights = tf.variance_scaling_initializer(
        scale=config.init_output_factor, mode='fan_in', distribution='normal')
    before_softplus_std_initializer = tf.constant_initializer(
        np.log(np.exp(config.init_std) - 1))

    input_ = tf.reshape(observations, shape=[batch_size, episode_len, observations.shape.as_list()[
                        2], functools.reduce(operator.mul, observations.shape.as_list()[3:], 1)])

    with tf.variable_scope('o_trunk'):
        conv1 = tf.layers.conv2d(
            inputs=input_,
            filters=64,
            kernel_size=[3, input_.shape.as_list()[-1]],
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3, conv1.shape.as_list()[-1]],
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        flatten = tf.reshape(conv2, shape=[batch_size, episode_len, functools.reduce(
            operator.mul, conv1.shape.as_list()[2:], 1)])
        trunk_fc = tf.layers.dense(
            inputs=flatten,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        with tf.variable_scope('o_crown'):
            # offensive
            off_fc = tf.layers.dense(
                inputs=trunk_fc,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=init_xavier_weights,
            )
            with tf.variable_scope('policy'):
                with tf.variable_scope('actions'):
                    with tf.variable_scope('powers'):
                        powers = tf.layers.dense(
                            inputs=off_fc,
                            units=5,
                            activation=tf.nn.relu,
                            kernel_initializer=init_output_weights,
                        )
                    with tf.variable_scope('directions'):
                        directions = tf.layers.dense(
                            inputs=off_fc,
                            units=6,
                            activation=tf.tanh,  # TODO tanh is not good?
                            kernel_initializer=init_output_weights,
                        )
                    off_actions_mean = tf.concat(
                        [directions[:, :, 0:1], powers, directions[:, :, 1:6]], axis=2)  # shape=(batch, len, 1+5+5)
                    off_actions_std = tf.nn.softplus(tf.get_variable(  # TODO
                        'before_softplus_std', off_actions_mean.shape[2:], tf.float32,
                        before_softplus_std_initializer))
                    off_actions_std = tf.tile(
                        off_actions_std[None, None],
                        [batch_size, episode_len, 1])
                    off_actions_mean = tf.check_numerics(
                        off_actions_mean, 'off_actions_mean')
                    off_actions_std = tf.check_numerics(
                        off_actions_std, 'off_actions_std')
                    off_actions = CustomKLDiagNormal(
                        off_actions_mean, off_actions_std)
                with tf.variable_scope('decision'):
                    logits = tf.layers.dense(
                        inputs=off_fc,
                        units=3,
                        activation=None,
                        kernel_initializer=init_output_weights,
                    )
                    off_decision = tfd.Categorical(logits)
                off_policy = [off_decision, off_actions]
            with tf.variable_scope('value'):
                off_value = tf.layers.dense(
                    inputs=off_fc,
                    units=1,
                    activation=None,
                    kernel_initializer=init_output_weights,
                )
                off_value = tf.reshape(off_value, shape=[batch_size, episode_len])
                off_value = tf.check_numerics(off_value, 'off_value')

    with tf.variable_scope('d_trunk'):
        conv1 = tf.layers.conv2d(
            inputs=input_,
            filters=64,
            kernel_size=[3, input_.shape.as_list()[-1]],
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3, conv1.shape.as_list()[-1]],
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        flatten = tf.reshape(conv2, shape=[batch_size, episode_len, functools.reduce(
            operator.mul, conv1.shape.as_list()[2:], 1)])
        trunk_fc = tf.layers.dense(
            inputs=flatten,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=init_xavier_weights,
        )
        with tf.variable_scope('d_crown'):
            # defensive
            def_fc = tf.layers.dense(
                inputs=trunk_fc,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=init_xavier_weights,
            )
            with tf.variable_scope('policy'):
                with tf.variable_scope('actions'):
                    with tf.variable_scope('powers'):
                        powers = tf.layers.dense(
                            inputs=def_fc,
                            units=5,
                            activation=tf.nn.relu,
                            kernel_initializer=init_output_weights,
                        )
                    with tf.variable_scope('directions'):
                        directions = tf.layers.dense(
                            inputs=def_fc,
                            units=5,
                            activation=tf.tanh,  # TODO tanh is not good?
                            kernel_initializer=init_output_weights,
                        )
                    def_actions_mean = tf.concat(
                        [powers, directions], axis=2)  # shape=[batch, len, 5+5]
                    def_actions_std = tf.nn.softplus(tf.get_variable(  # TODO
                        'before_softplus_std', def_actions_mean.shape[2:], tf.float32,
                        before_softplus_std_initializer))
                    def_actions_std = tf.tile(
                        def_actions_std[None, None],
                        [batch_size, episode_len, 1])
                    def_actions_mean = tf.check_numerics(
                        def_actions_mean, 'def_actions_mean')
                    def_actions_std = tf.check_numerics(
                        def_actions_std, 'def_actions_std')
                    def_actions = CustomKLDiagNormal(
                        def_actions_mean, def_actions_std)
                def_policy = def_actions
            with tf.variable_scope('value'):
                def_value = tf.layers.dense(
                    inputs=def_fc,
                    units=1,
                    activation=None,
                    kernel_initializer=init_output_weights,
                )
                def_value = tf.reshape(def_value, shape=[batch_size, episode_len])
                def_value = tf.check_numerics(def_value, 'def_value')
                

    policy = off_policy + [def_policy]
    value = [off_value, def_value]
    return agents.tools.AttrDict(state=state, policy=policy, value=value)
