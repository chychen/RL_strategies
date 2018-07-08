
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
    """Constructor for an instance of the environment.

    Args
    -----
    config : Object providing configurations via attributes.
    outdir: Directory to store videos in.

    Returns
    -------
    Wrapped OpenAI Gym environment.
    """
    env = gym.make(config.env)
    env = BBallWrapper(env, data=h5py.File('bball_strategies/data/GAILTransitionData_{}.hdf5'.format(
        config.train_len), 'r'), fps=config.FPS, time_limit=config.max_length)
    return env


def _define_loop(graph, logdir, train_steps, eval_steps):
    """Create and configure a training loop with training and evaluation phases.

    Args:
      graph: Object providing graph elements via attributes.
      logdir: Log directory for storing checkpoints and summaries.
      train_steps: Number of training steps per epoch.
      eval_steps: Number of evaluation steps per epoch.

    Returns:
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
        checkpoint_every=200 * eval_steps,
        feed={graph.is_training: False,
              graph.is_optimizing_offense: True})
    return loop


def test_policy(config, vanilla_env, steps, ppo_policy, D, denormalize_observ):
    """ test policy
    - draw episode into mpeg video
    - collect episdoe and each state scores into numpy
    """
    timer = time.time()
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
        vanilla_obs, _, _, _ = vanilla_env.step(
            vanilla_trans_act)
        numpy_collector.append(vanilla_obs)
    numpy_collector = np.array(numpy_collector)
    act_collector = np.array(act_collector)
    reward_collector = D.get_rewards_value(
        numpy_collector[None, :, -1], act_collector[None])
    numpy_collector = denormalize_observ(numpy_collector)
    np.savez(os.path.join(config.logdir, 'gail_testing_G{}_D{}/episode_{}.npz'.format(config.max_length, config.D_len, steps)),
             STATE=numpy_collector[:, -1], REWARD=reward_collector[0])
    print('test_policy time cost: {}'.format(time.time() - timer))


def tally_reward_line_chart(config, steps, ppo_policy, D, normalize_observ, normalize_action):
    """ tally 100 episodes as line chart to show how well the discriminator judge on each state of real and fake episode
    """
    timer = time.time()
    episode_amount = 100
    # env
    vanilla_env = gym.make(config.env)
    vanilla_env = BBallWrapper(vanilla_env, data=h5py.File('bball_strategies/data/GAILTransitionData_52.hdf5', 'r'), init_mode=1, fps=config.FPS, if_back_real=False,
                               time_limit=50)
    # fake
    numpy_collector = []
    act_collector = []
    real_numpy_collector = []
    real_act_collector = []
    for _ in range(episode_amount):
        vanilla_obs = vanilla_env.reset()
        epi_obs = []
        epi_act = []
        real_epi_obs = []
        real_epi_act = []
        for _ in range(vanilla_env.time_limit):
            vanilla_act = ppo_policy.act(
                np.array(vanilla_obs)[None, None], stochastic=False)
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
            epi_obs.append(vanilla_obs[-1])
            epi_act.append(vanilla_act.reshape([5, 2]))
            real_epi_obs.append(np.concatenate(
                [vanilla_obs[-1, 0:6], normalize_observ(info['expert_s']), vanilla_obs[-1, 11:14]], axis=0))
            real_epi_act.append(info['expert_a'])
        numpy_collector.append(epi_obs)
        act_collector.append(epi_act)
        real_numpy_collector.append(real_epi_obs)
        real_act_collector.append(real_epi_act)
    numpy_collector = np.array(numpy_collector)
    act_collector = np.array(act_collector)
    real_numpy_collector = np.array(real_numpy_collector)
    real_act_collector = np.array(real_act_collector)
    fake_rewards = D.get_rewards_value(
        numpy_collector, act_collector)
    real_act_collector = normalize_action(real_act_collector)
    real_rewards = D.get_rewards_value(
        real_numpy_collector, real_act_collector)
    # vis
    vis_line_chart(real_rewards, fake_rewards, config.logdir, str(steps))
    print('tally_reward_line_chart time cost: {}'.format(time.time() - timer))


def capped_video_schedule(episode_id):
    return episode_id % 500 == 0


class MonitorWrapper(gym.wrappers.Monitor):
    # init_mode 0 : init by default
    def __init__(self, env, init_mode=None, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, init_ball_handler_idx=None, directory='./test/', if_back_real=False, video_callable=capped_video_schedule):
        super(MonitorWrapper, self).__init__(env=env, directory=directory,
                                             video_callable=video_callable, force=True)
        self._env = env
        self._env.init_mode = init_mode
        self._env.if_vis_trajectory = if_vis_trajectory
        self._env.if_vis_visual_aid = if_vis_visual_aid
        self._env.init_positions = init_positions
        self._env.init_ball_handler_idx = init_ball_handler_idx
        self._env.if_back_real = if_back_real

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def data(self):
        return self._env.data

    @data.setter
    def data(self, value):
        self._env.data = value


def train(config, env_processes, outdir):
    """ Training and evaluation entry point yielding scores.

    Resolves some configuration attributes, creates environments, graph, and
    training loop. By default, assigns all operations to the CPU.

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

    def normalize_observ(observ):
        min_ = dummy_env.observation_space.low[0, 0]
        max_ = dummy_env.observation_space.high[0, 0]
        observ = 2.0 * (observ - min_) / (max_ - min_) - 1.0
        return observ

    def normalize_action(act):
        min_ = dummy_env.action_space[3].low
        max_ = dummy_env.action_space[3].high
        act = 2.0 * (act - min_) / (max_ - min_) - 1.0
        return act

    def denormalize_observ(observ):
        min_ = dummy_env.observation_space.low[0]
        max_ = dummy_env.observation_space.high[0]
        observ = (observ + 1.0) * (max_ - min_) / 2.0 + min_
        return observ

    # env to testing
    vanilla_env = gym.make(config.env)
    vanilla_env = BBallWrapper(vanilla_env, data=h5py.File('bball_strategies/data/GAILTransitionData_52.hdf5', 'r'), init_mode=1, fps=config.FPS, if_back_real=False,
                               time_limit=50)
    vanilla_env = MonitorWrapper(vanilla_env, directory=os.path.join(config.logdir, 'gail_testing_G{}_D{}/'.format(config.max_length, config.D_len)), if_back_real=False, video_callable=lambda _: True,
                                 # init from dataset
                                 init_mode=1)
    # env to generate fake state
    # env = gym.make(config.env)
    # env = BBallWrapper(env, data=h5py.File('bball_strategies/data/GAILTransitionData_{}.hdf5'.format(config.train_len), 'r'), init_mode=1, fps=config.FPS, if_back_real=config.if_back_real,
    #                    time_limit=config.max_length)
    # env = MonitorWrapper(env, directory=os.path.join(config.logdir, 'gail_training/'), if_back_real=config.if_back_real,
    #                      # init from dataset in order
    #                      init_mode=1)
    # PPO graph
    if config.update_every % config.num_agents:
        tf.logging.warn('Number of agents should divide episodes per update.')
    with tf.device('/cpu:0'):
        batch_env = utility.define_batch_env(
            lambda: _create_environment(config),
            config.num_agents, env_processes, outdir=outdir, is_gail=config.is_gail)
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
    # summary writer of Discriminator
    summary_writer = tf.summary.FileWriter(config.logdir + '/Disciminator')
    # TF Session
    # TODO _num_finished_episodes => Variable:0
    saver = utility.define_saver(
        exclude=(r'.*_temporary.*', r'.*memory.*', r'Variable:0', r'.*Adam.*', r'.*beta.*'))
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=config.log_device_placement)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        utility.initialize_variables(
            sess, saver, config.logdir, resume=FLAGS.resume)
        # NOTE reset variables in optimizer
        opt_reset_D = tf.group(
            [v.initializer for v in graph.algo.D.optimizer.variables()])
        # reset PPO optimizer
        opt_reset = tf.group(
            [v.initializer for v in graph.algo._optimizer.variables()])
        sess.run([opt_reset, opt_reset_D])
        # visulization stuff
        if FLAGS.tally_only:
            tally_reward_line_chart(config, sess.run(
                graph.algo.D._steps), ppo_policy, D, normalize_observ, normalize_action)
            exit()
        # GAIL
        cumulate_steps = sess.run(graph.step)
        counter = 0
        while True:
            # train Discriminator
            gail_timer = time.time()
            if counter > config.pretrain_d_times:
                num_d_to_train = config.train_d_per_ppo
            else:
                num_d_to_train = config.pretrain_d_per_ppo
            for _ in range(num_d_to_train):
                # train D
                feed_dict = {
                    graph.is_training: True,
                    graph.should_log: True,
                    graph.do_report: True,
                    graph.force_reset: False}
                gail_counter = 0
                while gail_counter < config.gail_steps:
                    gail_summary = sess.run(
                        graph.gail_summary, feed_dict=feed_dict)
                    if gail_summary:
                        summary_writer.add_summary(
                            gail_summary, global_step=sess.run(graph.algo.D._steps))
                    gail_counter += 1
                # testing
                if counter % (config.vis_testing_freq) == 0:
                    test_policy(config, vanilla_env, sess.run(graph.algo.D._steps), ppo_policy,
                                graph.algo.D, denormalize_observ)
                if counter % (config.tally_line_chart_freq) == 0:
                    tally_reward_line_chart(config, sess.run(
                        graph.algo.D._steps), ppo_policy, graph.algo.D, normalize_observ, normalize_action)
                counter += 1
            print('Time Cost of Discriminator per Update: {}'.format(
                (time.time() - gail_timer) / num_d_to_train))
            # train ppo
            cumulate_steps += total_steps
            for score in loop.run(sess, saver, cumulate_steps):
                yield score
    batch_env.close()
    vanilla_env.close()
    env.close()


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

    # train ppo
    for score in train(config, FLAGS.env_processes, outdir):
        tf.logging.info('Score {}.'.format(score))


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', 'logdir/gail_defense',
        'Base directory to store logs.')
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
