
""" configuration to pretrain policy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from agents import algorithms
from bball_strategies import algorithms
from bball_strategies.networks import policy_net
from bball_strategies.algorithms import pretrain_model

FPS = 5


def default():
    """Default configuration for Pretrain."""
    # tensorflow
    log_device_placement = False
    # Network
    # weight_summaries = dict(
    #     all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
    init_output_factor = 0.1
    # Optimization
    optimizer = tf.train.AdamOptimizer
    learning_rate = 1e-3
    batch_size = 512
    num_epochs = int(1e4)
    checkpoint_every = 100
    # Loss
    loss_alpha = 0.5

    return locals()


def example():
    locals().update(default())
    return locals()


def offense():
    locals().update(default())
    network = policy_net.offense_pretrain
    # model = pretrain_model.PretrainOffense
    return locals()


def defense():
    locals().update(default())
    network = policy_net.defense_pretrain
    # model = pretrain_model.PretrainOffense
    return locals()
