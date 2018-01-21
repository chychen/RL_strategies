""" configuration using the PPO algorithm on BBall strategies"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def default():
    """ Default configuration for project. """
    # # TODO
    # # General
    # algorithm = ppo.PPOAlgorithm
    # num_agents = 30
    # eval_episodes = 30
    # use_gpu = True
    # # Network
    # network = networks.feed_forward_gaussian
    # weight_summaries = dict(
    #     all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
    # policy_layers = 200, 100
    # value_layers = 200, 100
    # init_mean_factor = 0.1
    # init_logstd = -1
    # # Optimization
    # update_every = 30
    # update_epochs = 25
    # optimizer = tf.train.AdamOptimizer
    # learning_rate = 1e-4
    # # Losses
    # discount = 0.995
    # kl_target = 1e-2
    # kl_cutoff_factor = 2
    # kl_cutoff_coef = 1000
    # kl_init_penalty = 1
    # # Environment
    # env = 'Pendulum-v0'
    # max_length = 200
    # steps = 2e6  # 2M

    return locals()

def bball_strategies():
    locals().update(default())
    return locals()