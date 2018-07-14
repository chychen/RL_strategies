
""" Script to train
Command Line :
    train : python3 -m bball_strategies.scripts.train --config={function name}
    retrain : python3 -m bball_strategies.scripts.train --resume --logdir={checkpoint/dir/path}
    vis : python3 -m bball_strategies.scripts.train --vis
    tally_only : python3 -m bball_strategies.scripts.gail.train --config=episode_len_21 --resume --tally_only --logdir=logdir/gail_defense/20180530T160131-episode_len_11
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import gym
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.python import debug as tf_debug

from agents import tools
from agents.scripts import utility
from bball_strategies import gym_bball
from bball_strategies.scripts.gail import configs
from bball_strategies.gym_bball.tools import BBallWrapper
from bball_strategies.algorithms.discriminator import Discriminator
from bball_strategies.algorithms.ppo_generator import PPOPolicy
from bball_strategies.analysis.tally_discriminator_reward import vis_line_chart


def _create_environment(config):
    """ Constructor for an instance of the environment.

    Args
    -----
    config : Object providing configurations via attributes.
    outdir: Directory to store videos in.

    Returns
    -------
    Wrapped OpenAI Gym environment.
    """
    env = gym.make(config.env)
    env = BBallWrapper(env, data=h5py.File('bball_strategies/data/OrderedGAILTransitionData_{}.hdf5'.format(
        config.train_len), 'r'), fps=config.FPS, time_limit=config.max_length)
    return env


def _define_loop(graph, logdir, train_steps, eval_steps):
    """ Create and configure a training loop with training and evaluation phases.

    Args
    -----
    graph : Object providing graph elements via attributes.
    logdir : Log directory for storing checkpoints and summaries.
    train_steps : Number of training steps per epoch.
    eval_steps : Number of evaluation steps per epoch.

    Returns
    -------
    Loop object.
    """
    loop = tools.Loop(
        logdir, graph.step, graph.should_log, graph.do_report,
        graph.force_reset)
    loop.add_phase(
        'train_offense', graph.done, graph.score, graph.summary, train_steps,
        report_every=train_steps,
        log_every=train_steps,
        checkpoint_every=None,
        feed={graph.is_training: True,
              graph.is_optimizing_offense: True})
    loop.add_phase(
        'eval_offense', graph.done, graph.score, graph.summary, eval_steps,
        report_every=eval_steps,
        log_every=eval_steps,
        checkpoint_every=150 * eval_steps,
        feed={graph.is_training: False,
              graph.is_optimizing_offense: True})
    return loop


def collect_results(config, steps, ppo_policy, D, denormalize_observ, generated_amount=10):
    """ test policy
    - draw episode into mpeg video
    - collect episode with scores on each frame into .npz file (for out customized player)

    Args
    -----
    config : object, providing configurations via attributes.
    vanilla_env : object, env
    steps : int, to name the file with number of iterations of Discriminator
    ppo_policy : object, policy to generate actions
    D : object, discriminator to judge realistic
    denormalize_observ : function, denorm the returned observation
    """
    timer = time.time()
    # read condition length
    data_len = np.load('bball_strategies/data/FixedFPS5Length.npy')
    # env to testing
    vanilla_env = gym.make(config.env)
    vanilla_env = BBallWrapper(vanilla_env, data=h5py.File(
        'bball_strategies/data/OrderedGAILTransitionData_Testing.hdf5', 'r'), init_mode=1, fps=config.FPS, time_limit=np.max(data_len)-2)
    vanilla_env = MonitorWrapper(vanilla_env, directory=os.path.join(config.logdir, 'collect_result/video/'), video_callable=lambda _: True,
                                 # init from dataset
                                 init_mode=1)
    total_output = []
    index_list = []
    for i in range(generated_amount):
        print('generating # {} episode'.format(i))
        numpy_collector = []
        act_collector = []
        vanilla_obs = vanilla_env.reset()
        for _ in range(vanilla_env.time_limit):
            vanilla_act = ppo_policy.act(
                np.array(vanilla_obs)[None, None], stochastic=False)
            act_collector.append(vanilla_act.reshape([5, 2]))
            vanilla_trans_act = [
                # Discrete(3) must be int
                int(0),
                # Box(2,)
                np.array([0.0, 0.0], dtype=np.float32),
                # Box(5, 2)
                np.zeros(shape=[5, 2], dtype=np.float32),
                # Box(5, 2)
                np.reshape(vanilla_act, [5, 2])
            ]
            vanilla_obs, _, _, info = vanilla_env.step(
                vanilla_trans_act)
            numpy_collector.append(vanilla_obs)
            print(info['data_idx'])
        index_list.append(info['data_idx'])
        numpy_collector = np.array(numpy_collector)
        act_collector = np.array(act_collector)
        numpy_collector = denormalize_observ(numpy_collector)
        total_output.append(numpy_collector)
    total_output = np.array(total_output)
    # save numpy
    np.save(os.path.join(config.logdir,
                         'collect_result/total_output.npy'), total_output)
    np.save(os.path.join(config.logdir,
                         'collect_result/total_output_length.npy'), data_len[index_list]-2)

    print('collect_results time cost: {} per episode'.format(
        (time.time() - timer)/generated_amount))
    vanilla_env.close()


class MonitorWrapper(gym.wrappers.Monitor):
    """ class wrapper to record the interaction between policy and environment
    """

    def __init__(self, env, init_mode=None, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, init_ball_handler_idx=None, directory='./test/', video_callable=lambda _: True):
        super(MonitorWrapper, self).__init__(env=env, directory=directory,
                                             video_callable=video_callable, force=True)
        self._env = env
        self._env.init_mode = init_mode
        self._env.if_vis_trajectory = if_vis_trajectory
        self._env.if_vis_visual_aid = if_vis_visual_aid
        self._env.init_positions = init_positions
        self._env.init_ball_handler_idx = init_ball_handler_idx

    def __getattr__(self, name):
        return getattr(self._env, name)


def testing(config, env_processes, outdir):
    """ testing

    Args
    ----
    config : Object providing configurations via attributes.
    env_processes : Whether to step environment in external processes.
    outdir : Directory path to save rendering result while traning.

    Yields
    ------
    score : Evaluation scores.
    """
    tf.reset_default_graph()
    # env to get config
    dummy_env = gym.make(config.env)

    def denormalize_observ(observ):
        min_ = dummy_env.observation_space.low[0]
        max_ = dummy_env.observation_space.high[0]
        observ = (observ + 1.0) * (max_ - min_) / 2.0 + min_
        return observ

    # PPO graph
    if config.update_every % config.num_agents:
        tf.logging.warn('Number of agents should divide episodes per update.')
    with tf.device('/cpu:0'):
        batch_env = utility.define_batch_env(
            lambda: _create_environment(config),
            config.num_agents, env_processes, outdir=outdir)
        graph = utility.define_simulation_graph(
            batch_env, config.algorithm, config)
        loop = _define_loop(
            graph, config.logdir,
            config.update_every * config.max_length,
            config.eval_episodes * config.max_length)
        total_steps = int(
            config.steps / config.update_every *
            (config.update_every + config.eval_episodes))
    # Agent to genrate acttion
    ppo_policy = PPOPolicy(config, dummy_env)
    # TF Session
    # NOTE: _num_finished_episodes => Variable:0
    saver = utility.define_saver(
        exclude=(r'.*_temporary.*', r'.*memory.*', r'Variable:0'))
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=config.log_device_placement)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        utility.initialize_variables(
            sess, saver, config.logdir, checkpoint=FLAGS.checkpoint, resume=FLAGS.resume)
        # testing
        collect_results(config, sess.run(graph.algo.D._steps), ppo_policy,
                        graph.algo.D, denormalize_observ)
    batch_env.close()


def main(_):
    """ Create or load configuration and launch the trainer.
    """
    utility.set_up_logging()
    if FLAGS.resume:
        logdir = FLAGS.logdir
    else:
        logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
            FLAGS.logdir, '{}-{}'.format(FLAGS.timestamp, FLAGS.config)))
    if FLAGS.vis:
        outdir = os.path.join(logdir, 'train_output')
    else:
        outdir = None

    if not FLAGS.config:
        raise KeyError('You must specify a configuration.')
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)

    # collecting
    testing(config, FLAGS.env_processes, outdir)


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', 'logdir/gail_defense',
        'Base directory to store logs.')
    tf.app.flags.DEFINE_string(
        'checkpoint', None,
        'Checkpoint name to load; defaults to most recent.')
    tf.app.flags.DEFINE_string(
        'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
        'Sub directory to store logs.')
    tf.app.flags.DEFINE_string(
        'config', None,
        'Configuration to execute.')
    tf.app.flags.DEFINE_boolean(
        'env_processes', True,
        'Step environments in separate processes to circumvent the GIL.')
    tf.app.flags.DEFINE_boolean(
        'vis', False,
        'whether to vis during training')
    tf.app.flags.DEFINE_boolean(
        'resume', False,
        'whether to resume training')
    tf.app.flags.DEFINE_boolean(
        'tally_only', False,
        'whether to tally the reward line chart')
    tf.app.run()
