
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


def norm_obs(env, data):
    min_ = env.observation_space.low
    max_ = env.observation_space.high
    data = 2 * (data - min_) / (max_ - min_) - 1
    return data


def pack_action(actions, team):
    if team == 'offense':
        decision_logits, action = actions
        decision = np.argmax(decision_logits.reshape([3, ]))
        ret_action = tuple((
            np.array(decision),
            action[:2],
            action[2:].reshape([5, 2]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
    elif team == 'defense':
        ret_action = tuple((
            np.array(0),
            np.array([0, 0]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            actions.reshape([5, 2])
        ))
    return ret_action


def vis_data(off_data, off_label, def_data, def_label, outdir, start_idx=0):
    """ vis the dataset of transitions

    Args
    ----
    """

    idx_ = start_idx

    init_pos = [
        np.array(off_data[idx_, 0, -1, 0, :]),
        np.array(off_data[idx_, 0, -1, 1:6, :], dtype=np.float),
        np.array(off_data[idx_, 0, -1, 6:11, :], dtype=np.float)
    ]
    env = gym.make('bball-pretrain-v0')
    env = BBallWrapper(env, if_clip=False, if_norm_obs=False, if_norm_act=False, init_mode=2, if_vis_visual_aid=True,
                       if_vis_trajectory=False, init_positions=init_pos)
    env = gym.wrappers.Monitor(
        env, outdir, lambda unused_episode_number: True, force=False, resume=True)
    obs = env.reset()

    while True:
        # prevent from modification
        temp_off_label = np.array(off_label[idx_, 0])
        temp_def_label = np.array(def_label[idx_, 0])
        if idx_ == start_idx:
            # the env's velocity is zero, so we add the last velocity after env reset.
            last_vel = off_data[idx_, 0, -1, 1:6, :] - \
                off_data[idx_, 0, -2, 1:6, :]
            temp_off_label[5:] += last_vel.reshape([10, ])
            last_vel = def_data[idx_, 0, -1, 6:11, :] - \
                def_data[idx_, 0, -2, 6:11, :]
            temp_def_label += last_vel.reshape([10, ])
        # offense
        action = pack_action(
            [temp_off_label[:3], temp_off_label[3:]], team='offense')
        obs, _, done, _ = env.step(action)
        if done:
            env.close()
            break
        # deffense
        action = pack_action(temp_def_label, team='defense')
        obs, _, done, _ = env.step(action)
        if done:
            env.close()
            break
        idx_ += 1


def vis_result(sess, model, off_data, off_label, def_data, def_label, outdir, num_video):
    """ vis the results by using the pretrain output interacting with env 

    Args
    ----
    """

    for i in range(num_video):
        start_idx = i * 50
        idx_ = start_idx
        init_pos = [
            np.array(off_data[idx_, 0, -1, 0, :]),
            np.array(off_data[idx_, 0, -1, 1:6, :], dtype=np.float),
            np.array(off_data[idx_, 0, -1, 6:11, :], dtype=np.float)
        ]
        env = gym.make('bball-pretrain-v0')
        env = BBallWrapper(env, if_clip=False, if_norm_obs=False, if_norm_act=False, init_mode=2, if_vis_visual_aid=True,
                           if_vis_trajectory=False, init_positions=init_pos)
        env = gym.wrappers.Monitor(
            env, outdir, lambda unused_episode_number: True, force=False, resume=True)
        obs = env.reset()

        while True:
            # prevent from modification
            temp_off_label = np.array(off_label[idx_, 0])
            temp_def_label = np.array(def_label[idx_, 0])
            if idx_ == start_idx:
                # the env's velocity is zero, so we add the last velocity after env reset.
                last_vel = off_data[idx_, 0, -1, 1:6, :] - \
                    off_data[idx_, 0, -2, 1:6, :]
                temp_off_label[5:] += last_vel.reshape([10, ])
                last_vel = def_data[idx_, 0, -1, 6:11, :] - \
                    def_data[idx_, 0, -2, 6:11, :]
                temp_def_label += last_vel.reshape([10, ])
            if FLAGS.config == 'offense':
                # offense turn
                obs = norm_obs(env, obs)
                logits, actions = model.perform(sess, obs[None, None])
                actions = pack_action(
                    [logits[0, 0], actions[0, 0]], FLAGS.config)
                obs, _, done, _ = env.step(actions)
                if done:
                    env.close()
                    break
                # defense turn
                actions = pack_action(temp_def_label, team='defense')
                obs, _, done, _ = env.step(actions)
                if done:
                    env.close()
                    break
            elif FLAGS.config == 'defense':
                # offense turn
                actions = pack_action(
                    [temp_off_label[:3], temp_off_label[3:]], team='offense')
                obs, _, done, _ = env.step(actions)
                if done:
                    env.close()
                    break
                # defense turn
                obs = norm_obs(env, obs)
                actions = model.perform(sess, obs[None, None])
                actions = pack_action(actions, FLAGS.config)
                obs, _, done, _ = env.step(actions)
                if done:
                    env.close()
                    break
            idx_ += 1


def testing(config, off_data, off_label, def_data, def_label, outdir):
    """

    Args
    ----
    config : Object providing configurations via attributes.

    Yields
    ------
    score : Evaluation scores.
    """
    # split into train and eval
    off_train_data, off_eval_data = np.split(
        off_data, [off_data.shape[0]*9//10])
    off_train_label, off_eval_label = np.split(
        off_label, [off_data.shape[0]*9//10])
    def_train_data, def_eval_data = np.split(
        def_data, [def_data.shape[0]*9//10])
    def_train_label, def_eval_label = np.split(
        def_label, [def_data.shape[0]*9//10])
    print(off_train_data.shape)
    print(off_eval_data.shape)
    print(off_train_label.shape)
    print(off_eval_label.shape)
    print(def_train_data.shape)
    print(def_eval_data.shape)
    print(def_train_label.shape)
    print(def_eval_label.shape)

    # graph
    tf.reset_default_graph()
    if FLAGS.config == 'offense':
        model = models.PretrainOffense(config)
    elif FLAGS.config == 'defense':
        model = models.PretrainDefense(config)
    else:
        raise ValueError('{} is not an available config'.format(FLAGS.config))

    message = 'Graph contains {} trainable variables.'
    tf.logging.info(message.format(tools.count_weights()))
    saver = utility.define_saver()
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=config.log_device_placement)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(
                sess, ui_type=FLAGS.ui_type)
        utility.initialize_variables(
            sess, saver, config.logdir, resume=True)
        vis_result(sess, model, off_train_data, off_train_label,
                   def_train_data, def_train_label, os.path.join(outdir, 'train'), 3)
        vis_result(sess, model, off_eval_data, off_eval_label,
                   def_eval_data, def_eval_label, os.path.join(outdir, 'eval'), 3)


def main(_):
    """ Create or load configuration and launch the trainer.
    """
    off_data = np.load('bball_strategies/pretrain/data/off_obs.npy')
    off_label = np.load('bball_strategies/pretrain/data/off_actions.npy')
    def_data = np.load('bball_strategies/pretrain/data/def_obs.npy')
    def_label = np.load('bball_strategies/pretrain/data/def_actions.npy')

    utility.set_up_logging()

    logdir = FLAGS.logdir
    outdir = os.path.expanduser(os.path.join(
        FLAGS.logdir, 'vis-{}-{}'.format(FLAGS.timestamp, FLAGS.config)))
    # config = utility.load_config(logdir)
    # try:
    #     config = utility.load_config(logdir)
    # except IOError:
    #     if not FLAGS.config:
    #         raise KeyError('You must specify a configuration.')
    #     config = tools.AttrDict(getattr(configs, FLAGS.config)())
    #     config = utility.save_config(config, logdir)
    if not FLAGS.config:
        raise KeyError('You must specify a configuration.')
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)
    
    vis_data(off_data, off_label, def_data, def_label, outdir, start_idx=0)
    testing(config, off_data, off_label, def_data, def_label, outdir)


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
        'debug', False,
        'whether to enable debug mode')
    tf.app.flags.DEFINE_string(
        'ui_type', 'curses',
        "Command-line user interface type (curses | readline)")
    tf.app.run()
