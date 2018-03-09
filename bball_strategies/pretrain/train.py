
""" Script to train
Command Line:
    python3 -m bball_strategies.scripts.train
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import gym
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from agents import tools
from agents.scripts import utility
from bball_strategies.gym_bball.envs.bball_pretrain_env import BBallPretrainEnv
from bball_strategies.scripts.bball_env_wrapper import BBallWrapper
from bball_strategies.pretrain import configs
from bball_strategies.pretrain import models


def training(sess, model, data, label, config, writter):
    """
    """
    # shuffle
    idx = np.random.permutation(len(data))
    data, label = data[idx], label[idx]
    num_batch = data.shape[0] // config.batch_size
    total_loss = 0
    for batch_idx in range(num_batch):
        data_idx = batch_idx * config.batch_size
        b_data = data[data_idx:data_idx + config.batch_size]
        b_label = label[data_idx:data_idx + config.batch_size]
        summary, loss, steps = model.train(sess, b_data, b_label)
        total_loss = total_loss + loss
        writter.add_summary(summary, global_step=steps)
    tf.logging.info('Training Mean Loss {}.'.format(total_loss/num_batch))


def evaluating(sess, model, data, label, config, writter):
    """
    """
    summary, loss, steps = model.eval(sess, data, label)
    writter.add_summary(summary, global_step=steps)
    tf.logging.info('Evaluating Mean Loss {}.'.format(loss))
    # num_batch = data.shape[0] // config.batch_size
    # total_loss = 0
    # for batch_idx in range(num_batch):
    #     data_idx = batch_idx * config.batch_size
    #     b_data = data[data_idx:data_idx + config.batch_size]
    #     b_label = label[data_idx:data_idx + config.batch_size]
    #     summary, loss, steps = model.eval(sess, b_data, b_label)
    #     total_loss = total_loss + loss
    #     # writter.add_summary(summary, global_step=steps)
    # tf.logging.info('Evaluating Mean Loss {}.'.format(total_loss/num_batch))


def train(config, data, label, outdir):
    """ Training and evaluation entry point yielding scores.

    Args
    ----
    config : Object providing configurations via attributes.

    Yields
    ------
    score : Evaluation scores.
    """
    # normalization
    env = BBallPretrainEnv()
    min_ = env.observation_space.low
    max_ = env.observation_space.high
    data = 2 * (data - min_) / (max_ - min_) - 1
    # split into train and eval
    train_data, eval_data = np.split(data, [data.shape[0]*9//10])
    train_label, eval_label = np.split(label, [data.shape[0]*9//10])
    print(train_data.shape)
    print(eval_data.shape)
    # graph
    tf.reset_default_graph()
    if FLAGS.config == 'offense':
        model = models.PretrainOffense(config)
    elif FLAGS.config == 'defense':
        model = models.PretrainDefense(config)
    else:
        raise ValueError('{} is not an available config'.format(FLAGS.config))
    # model = config.model(config)
    message = 'Graph contains {} trainable variables.'
    tf.logging.info(message.format(tools.count_weights()))
    saver = utility.define_saver()
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=config.log_device_placement)
    sess_config.gpu_options.allow_growth = True
    # summary writter
    train_writter = tf.summary.FileWriter(os.path.join(
        config.logdir, 'train'), tf.get_default_graph())
    # summary writter
    eval_writter = tf.summary.FileWriter(os.path.join(
        config.logdir, 'eval'), tf.get_default_graph())
    with tf.Session(config=sess_config) as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(
                sess, ui_type=FLAGS.ui_type)
        utility.initialize_variables(
            sess, saver, config.logdir, resume=FLAGS.resume)
        for epoch_idx in range(config.num_epochs):
            tf.logging.info('Number of epochs: {}'.format(epoch_idx))
            training(sess, model, train_data,
                     train_label, config, train_writter)
            evaluating(sess, model, eval_data,
                       eval_label, config, eval_writter)
            if (epoch_idx + 1) % config.checkpoint_every == 0:
                tf.gfile.MakeDirs(config.logdir)
                filename = os.path.join(config.logdir, 'model.ckpt')
                saver.save(sess, filename, (epoch_idx + 1) * config.batch_size)


def main(_):
    """ Create or load configuration and launch the trainer.
    """
    if FLAGS.config == 'offense':
        data = np.load('bball_strategies/pretrain/data/off_obs.npy')
        label = np.load('bball_strategies/pretrain/data/off_actions.npy')
    elif FLAGS.config == 'defense':
        data = np.load('bball_strategies/pretrain/data/def_obs.npy')
        label = np.load('bball_strategies/pretrain/data/def_actions.npy')
    else:
        raise ValueError('{} is not an available config'.format(FLAGS.config))
    utility.set_up_logging()
    if not FLAGS.resume:
        logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
            FLAGS.logdir, '{}-{}'.format(FLAGS.timestamp, FLAGS.config)))
    else:
        logdir = FLAGS.logdir
    if FLAGS.vis:
        outdir = os.path.join(logdir, 'train_output')
    else:
        outdir = None
    try:
        config = utility.load_config(logdir)
    except IOError:
        if not FLAGS.config:
            raise KeyError('You must specify a configuration.')
        config = tools.AttrDict(getattr(configs, FLAGS.config)())
        config = utility.save_config(config, logdir)
    train(config, data, label, outdir)


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', 'logdir/pretrain',
        'Base directory to store logs.')
    tf.app.flags.DEFINE_string(
        'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
        'Sub directory to store logs.')
    tf.app.flags.DEFINE_string(
        'config', None,
        'Configuration to execute.')
    tf.app.flags.DEFINE_boolean(
        'vis', False,
        'whether to vis during training')
    tf.app.flags.DEFINE_boolean(
        'debug', False,
        'whether to enable debug mode')
    tf.app.flags.DEFINE_boolean(
        'resume', False,
        'whether to restore checkpoint in logdir')
    tf.app.flags.DEFINE_string(
        'ui_type', 'curses',
        "Command-line user interface type (curses | readline)")
    tf.app.run()
