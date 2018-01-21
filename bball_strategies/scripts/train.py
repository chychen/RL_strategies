""" Script to train
Command Line:
    python3 -m TODO
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gym
from agents.scripts import utility
from agents import tools
from bball_strategies.scripts import configs
from bball_strategies import gym_bball

#TODO
#what is streaming estimation, and should we normalized the output?

def _create_environment(config):
    """Constructor for an instance of the environment.

    Args:
      config: Object providing configurations via attributes.

    Returns:
      Wrapped OpenAI Gym environment.
    """
    # TODO
    # if isinstance(config.env, str):
    #     env = gym.make(config.env)
    # else:
    #     env = config.env()
    # make bball strategies environment
    env = gym.make('bball-v0')
    # if config.max_length:
    #     env = tools.wrappers.LimitDuration(env, config.max_length)
    # env = tools.wrappers.RangeNormalize(env)
    # env = tools.wrappers.ClipAction(env)
    # env = tools.wrappers.ConvertTo32Bit(env)
    return env


def train(config):
    """ Training and evaluation entry point yielding scores.

    Resolves some configuration attributes, creates environments, graph, and
    training loop. By default, assigns all operations to the CPU.

    Args
    ----
        config : Object providing configurations via attributes.

    Yields
    ------
        score : Evaluation scores.
    """
    tf.reset_default_graph()
    if config.update_every % config.num_agents:
        tf.logging.warn('Number of agents should divide episodes per update.')
    with tf.device('/cpu:0'):
        batch_env = utility.define_batch_env(
            lambda: _create_environment(config)
        )


def main(_):
    """ Create or load configuration and launch the trainer.
    """
    utility.set_up_logging()
    if not FLAGS.config:
        raise (KeyError('You must specify a configuration.'))
    try:  # use original config file
        config = utility.load_config(FLAGS.logdir)
    except IOError:
        config = tools.AttrDict(getattr(configs, FLAGS.config)())
        config = utility.save_config(config, FLAGS.logdir)
    for score in train(config):  # score will be yielded from train()
        tf.logging.info('Score {}.'.format(score))


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', 'logdir/v1/1',
        'Base directory to store logs')
    tf.app.flags.DEFINE_string(
        'config', 'default',
        'Configuration to execute. (json format)')

    tf.app.run()
