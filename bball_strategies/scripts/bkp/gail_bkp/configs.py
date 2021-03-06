
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
    # optimizer = tf.train.AdamOptimizer
    # optimizer = tf.train.MomentumOptimizer
    optimizer = tf.train.GradientDescentOptimizer
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
    max_length = 11-1
    steps = update_every*max_length
    # wgan
    wgan_penalty_lambda = 10.0
    episodes_per_batch = 5
    d_batch_size = max_length * episodes_per_batch
    train_d_per_ppo = 5
    # Gail
    is_gail = False

    return locals()

def episode_len_11():
    locals().update(default())
    train_len = 11
    max_length = train_len-1
    # ppo
    update_every = 100
    steps = update_every*max_length
    # wgan
    episodes_per_batch = 20
    d_batch_size = max_length * episodes_per_batch
    return locals()

def episode_len_21():
    locals().update(default())
    train_len = 21
    max_length = train_len-1
    # ppo
    update_every = 100
    steps = update_every*max_length
    # wgan
    episodes_per_batch = 10
    d_batch_size = max_length * episodes_per_batch
    return locals()

def episode_len_31():
    locals().update(default())
    train_len = 31
    max_length = train_len-1
    # ppo
    update_every = 100
    steps = update_every*max_length
    # wgan
    episodes_per_batch = 7
    d_batch_size = max_length * episodes_per_batch
    return locals()

def episode_len_41():
    locals().update(default())
    train_len = 41
    max_length = train_len-1
    # ppo
    update_every = 100
    steps = update_every*max_length
    # wgan
    episodes_per_batch = 5
    d_batch_size = max_length * episodes_per_batch
    return locals()

def episode_len_51():
    locals().update(default())
    train_len = 51
    max_length = train_len-1
    # ppo
    update_every = 100
    steps = update_every*max_length
    # wgan
    episodes_per_batch = 4
    d_batch_size = max_length * episodes_per_batch
    return locals()

def episode_len_11_D():
    locals().update(episode_len_11())
    # wgan
    episodes_per_batch = 128
    d_batch_size = episodes_per_batch
    return locals()

def episode_len_21_D():
    locals().update(episode_len_21())
    # wgan
    episodes_per_batch = 128
    d_batch_size = episodes_per_batch
    return locals()

def episode_len_31_D():
    locals().update(episode_len_31())
    # wgan
    episodes_per_batch = 128
    d_batch_size = episodes_per_batch
    return locals()

def episode_len_41_D():
    locals().update(episode_len_41())
    # wgan
    episodes_per_batch = 128
    d_batch_size = episodes_per_batch
    return locals()

def episode_len_51_D():
    locals().update(episode_len_51())
    # wgan
    episodes_per_batch = 128
    d_batch_size = episodes_per_batch
    return locals()

def server_40cpu():
    locals().update(default())
    num_agents = 40
    eval_episodes = 40
    return locals()
