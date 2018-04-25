
""" configuration using the PPO algorithm on BBall strategies"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from agents import algorithms
from bball_strategies import algorithms
from bball_strategies.networks import policy_net

FPS = 5


def default():
    """Default configuration for PPO."""
    # tensorflow
    log_device_placement = False
    # General
    algorithm = algorithms.NDEF_PPO
    num_agents = 20
    eval_episodes = 10
    use_gpu = True
    # Environment
#   normalize_ranges = True
    # Network
    network = policy_net.ndef_gaussian
    weight_summaries = dict(
        all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
    init_output_factor = 0.1
    init_std = 0.35
    # Optimization
    update_every = 40
    update_epochs = 25
    optimizer = tf.train.AdamOptimizer
    learning_rate = 1e-4
#   batch_size = 20
#   chunk_length = 50
    # Losses
    discount = 0.995
    kl_target = 1e-2
    kl_cutoff_factor = 2
    kl_cutoff_coef = 1000
    kl_init_penalty = 1
#   gae_lambda = None
    entropy_regularization = None
    # Environment
    env = 'bball-ndef-v0'
    max_length = 24 * FPS
    steps = 1e10  # 1M

    return locals()


def v1():
    locals().update(default())
    update_every = 100
    discount = 1.0

    return locals()


def v2():
    locals().update(default())
    update_every = 100
    discount = 1.0
    env = 'bball-ndef-bspeed-v0'

    return locals()


def v3():
    locals().update(default())
    update_every = 200
    discount = 1.0
    env = 'bball-ndef-bspeed-v0'

    return locals()
