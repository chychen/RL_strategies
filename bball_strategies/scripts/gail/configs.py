
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
    eval_episodes = 20
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
#   gae_lambda = None # TODO
    # entropy_regularization = 0.1  # TODO
    # Environment
    env = 'bball_gail_def-v0'
    # data only have 50 length, if we step 50 times, cant find 51-th conidtion in data
    max_length = 49
    # make transitions buffer back to real
    if_back_real = False
    steps_per_ppo_iter = update_every*max_length
    steps = steps_per_ppo_iter*15
    # wgan
    wgan_penalty_lambda = 10.0
    episodes_per_batch = 5
    d_batch_size = max_length * episodes_per_batch
    train_d_per_ppo = 5

    return locals()

def server_40cpu():
    locals().update(default())
    num_agents = 40
    eval_episodes = 40
    return locals()
