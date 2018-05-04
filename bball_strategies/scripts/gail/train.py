
""" Script to train
Command Line :
    train : python3 -m bball_strategies.scripts.train --config={function name}
    retrain : python3 -m bball_strategies.scripts.train --resume --logdir={checkpoint/dir/path}
    vis : python3 -m bball_strategies.scripts.train --vis
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import gym
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

from agents import tools
from agents.scripts import utility
from bball_strategies import gym_bball
from bball_strategies.scripts.gail import configs
from bball_strategies.gym_bball.tools import BBallWrapper
from bball_strategies.algorithms.discriminator import Discriminator
from bball_strategies.algorithms.ppo_generator import PPOPolicy
from tensorboard.plugins.beholder import Beholder

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
    env = BBallWrapper(env, fps=config.FPS, time_limit=config.max_length)
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
        checkpoint_every=10 * eval_steps,
        feed={graph.is_training: False,
              graph.is_optimizing_offense: True})
    return loop


def capped_video_schedule_100(episode_id):
    return episode_id % 100 == 0


def capped_video_schedule_10(episode_id):
    return episode_id % 10 == 0


class MonitorWrapper(gym.wrappers.Monitor):
    # init_mode 0 : init by default
    def __init__(self, env, init_mode=None, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, init_ball_handler_idx=None, directory='./test/', if_back_real=True, video_callable=capped_video_schedule_10):
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

    dummy_env = gym.make(config.env)
    vanilla_env = gym.make(config.env)

    def normalize_observ(observ):
        min_ = dummy_env.observation_space.low
        max_ = dummy_env.observation_space.high
        observ = 2 * (observ - min_) / (max_ - min_) - 1
        return observ

    vanilla_env = BBallWrapper(vanilla_env, init_mode=1, fps=config.FPS, if_back_real=False,
                               time_limit=config.max_length)
    vanilla_env = MonitorWrapper(vanilla_env, directory=os.path.join(config.logdir, 'gail_episode/'), if_back_real=False,
                                 # init from dataset
                                 init_mode=1)

    tf.reset_default_graph()
    D = Discriminator(config, gym.make(config.env))
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
    # Exclude episode related variables since the Python state of environments is
    # not checkpointed and thus new episodes start after resuming.
    saver = utility.define_saver(exclude=(r'.*_temporary.*',))
    sess_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=config.log_device_placement)
    sess_config.gpu_options.allow_growth = True
    # env to generate fake state
    env = gym.make(config.env)
    # init_mode=3 : init from dataset in order
    env = BBallWrapper(env, init_mode=3, fps=config.FPS,
                       time_limit=config.max_length)
    # env = MonitorWrapper(env,directory=os.path.join(config.logdir, 'gail_state/'),
    #                      # init from dataset in order
    #                      init_mode=3)
    # agent to genrate acttion
    ppo_policy = PPOPolicy(config, env)
    beholder = Beholder(config.logdir)
    with tf.Session(config=sess_config) as sess:
        utility.initialize_variables(
            sess, saver, config.logdir, resume=FLAGS.resume)
        # GAIL
        expert_data = np.load('bball_strategies/data/GAILTransitionData.npy')
        print(expert_data.shape)
        cumulate_steps = sess.run(graph.step)
        while True:
            perm_idx = np.random.permutation(expert_data.shape[0])
            expert_data = expert_data[perm_idx]
            episode_idx = 0

            while episode_idx < expert_data.shape[0]-config.episodes_per_batch*config.train_d_per_ppo:
                beholder.update(session=sess)
                # testing
                vanilla_obs = vanilla_env.reset()
                for _ in range(config.max_length):
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
                    vanilla_obs, _, _, _ = vanilla_env.step(
                        vanilla_trans_act)
                # train Discriminator
                for _ in range(config.train_d_per_ppo):
                    print('train Discriminator')
                    batch_fake_states = []
                    batch_real_states = expert_data[episode_idx:episode_idx +
                                                    config.episodes_per_batch, 1:]  # frame 0 is condition
                    batch_real_states = np.concatenate(
                        batch_real_states, axis=0)
                    for _ in range(config.episodes_per_batch):
                        # align the conditions with env
                        # -1 : newest state
                        conditions = expert_data[episode_idx:episode_idx+1, :, :]
                        env.data = conditions[:, :, -1]
                        _ = env.reset()
                        for len_idx in range(config.max_length):
                            act = ppo_policy.act(
                                np.array(conditions[:, len_idx:len_idx+1, :]))
                            transformed_act = [
                                # Discrete(3) must be int
                                int(0),
                                # Box(2,)
                                np.array([0.0, 0.0], dtype=np.float32),
                                # Box(5, 2)
                                np.zeros(shape=[5, 2], dtype=np.float32),
                                # Box(5, 2)
                                np.reshape(act, [5, 2])
                            ]
                            obs_state, _, _, _ = env.step(
                                transformed_act)
                            batch_fake_states.append(obs_state)
                        episode_idx += 1
                    assert batch_real_states.shape[0] == len(batch_fake_states), "real: {}, fake: {}".format(
                        batch_real_states.shape[0], len(batch_fake_states))
                    batch_fake_states = np.array(batch_fake_states)
                    batch_real_states = normalize_observ(batch_real_states)
                    D.train(batch_fake_states, batch_real_states)
                # train PPO
                print('train PPO')
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
    tf.app.run()
