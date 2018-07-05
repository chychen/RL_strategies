
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
    # optimizer = tf.train.MomentumOptimizer
    # optimizer = tf.train.GradientDescentOptimizer
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
    # env = 'bball_gail_speed_def-v0'
    # data only have 50 length, if we step 50 times, cant find 51-th conidtion in data
    max_length = 11 - 1
    # make transitions buffer back to real
    if_back_real = False
    steps = update_every * max_length
    # wgan
    wgan_penalty_lambda = 10.0
    episodes_per_batch = 5
    d_batch_size = max_length * episodes_per_batch
    train_d_per_ppo = 5
    # Gail
    is_gail = False # to be depregated
    is_double_curiculum = False  # to be depregated

    return locals()

def double_curiculum():
    locals().update(default())
    ########################################################
    is_double_curiculum = True
    train_len = 12
    max_length = train_len - 2
    D_len = 10
    use_padding = False
    ########################################################
    num_agents = 20
    # ppo
    update_every = 100
    # no need to divide num_agent because they maintain steps made in class Loop
    steps = update_every * max_length
    # wgan
    train_d_per_ppo = 3
    pretrain_d_times = 30
    # one episode can generate 'max_length' episodes, d_batch_size must be the multiple of num_agents
    d_batch_size = 200
    if use_padding:
        episodes_per_batch = d_batch_size // (max_length - D_len + 10)
    else:
        if max_length == D_len:
            episodes_per_batch = d_batch_size
        else:
            episodes_per_batch = d_batch_size // (max_length - D_len)

    gail_steps = episodes_per_batch * max_length // num_agents
    return locals()
