# read tensor directly from the graph, then we could inference both actions and value
# and the returned action will interact with env in cotroller scripts to get observation -> fake samples to train with discriminator

import tensorflow as tf
from bball_strategies.networks import ppo_nets
from bball_strategies.scripts.no_defense import configs
import agents


# observation space = shape=(batch_size, episode_length, 10, 14, 2)
obs_dummy = tf.zeros(shape=(100, 100, 5, 14, 2))
# action space = shape=(batch, episode_length, 23)

with tf.variable_scope('ppo_generator'):
    # tf.get_variable_scope().reuse_variables()
    dict_ = ppo_nets.ndef_gaussian(agents.tools.AttrDict(configs.v3()), None, obs_dummy, None)
    print(dict_.state)
    print(dict_.policy)
    print(dict_.value)
    exit()