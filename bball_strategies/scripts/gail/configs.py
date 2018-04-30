
""" configuration using the PPO algorithm on BBall strategies"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from agents import algorithms
from bball_strategies import algorithms
from bball_strategies.networks import gail_ppo_nets
from bball_strategies.networks import discriminator_net

FPS = 5


def default():
    """Default configuration for PPO."""
    # tensorflow
    log_device_placement = False
    # General
    algorithm = algorithms.GAIL_DEF_PPO
    num_agents = 20
    eval_episodes = 10
    use_gpu = True
    # Environment
#   normalize_ranges = True
    # Network
    network = gail_ppo_nets.gail_def_gaussian
    d_network = discriminator_net.network
    weight_summaries = dict(
        all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
    init_output_factor = 0.1
    init_std = 0.35
    # Optimization
    update_every = 200
    update_epochs = 25
    optimizer = tf.train.AdamOptimizer
    learning_rate = 1e-4
#   batch_size = 20
#   chunk_length = 50
    # Losses
    discount = 1.0
    kl_target = 1e-2
    kl_cutoff_factor = 2
    kl_cutoff_coef = 1000
    kl_init_penalty = 1
#   gae_lambda = None # TODO
    # entropy_regularization = 0.1 # TODO
    # Environment
    env = 'bball_gail_def-v0'
    steps = 1e10  # 1M
    max_length = 50
    # wgan
    wgan_penalty_lambda = 10.0

    return locals()

