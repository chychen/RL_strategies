
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


def train_Discriminator(episode_idx, config, expert_data, expert_action, env, ppo_policy, D, normalize_observ, normalize_action):
    episode_counter = episode_idx
    for _ in range(config.train_d_per_ppo):
        print('train Discriminator')
        batch_fake_states = []
        fake_action = []
        batch_real_states = expert_data[episode_counter:episode_counter +
                                        config.episodes_per_batch, 1:]  # frame 0 is condition
        real_action = expert_action[episode_counter:
                                    episode_counter + config.episodes_per_batch, :-1]
        batch_real_states = np.concatenate(
            batch_real_states, axis=0)
        real_action = np.concatenate(real_action, axis=0)
        for _ in range(config.episodes_per_batch):
            # align the conditions with env
            # -1 : newest state
            conditions = expert_data[episode_counter:episode_counter+1, :, :]
            env.data = conditions[:, :, -1]
            obs_state = env.reset()
            for len_idx in range(config.max_length):
                if config.if_back_real:
                    act = ppo_policy.act(
                        np.array(conditions[:, len_idx:len_idx+1, :]), stochastic=True)
                else:
                    act = ppo_policy.act(
                        np.array(obs_state)[None, None], stochastic=True)
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
                fake_action.append(act.reshape([5, 2]))
            episode_counter += 1
        batch_fake_states = np.array(batch_fake_states)
        fake_action = np.array(fake_action)
        batch_real_states = normalize_observ(batch_real_states)
        # assert np.all(np.abs(batch_real_states[:, :, :6] - batch_fake_states[:, :, :6])<1e-2)
        assert batch_real_states.shape[0] == batch_fake_states.shape[0], "real: {}, fake: {}".format(
            batch_real_states.shape[0], batch_fake_states.shape[0])
        assert real_action.shape[0] == fake_action.shape[0], "real: {}, fake: {}".format(
            real_action.shape[0], fake_action.shape[0])
        D.train(batch_fake_states, batch_real_states,
                fake_action, normalize_action(real_action))


def valid_Discriminator(episode_idx, config, expert_data, expert_action, env, ppo_policy, D, normalize_observ, normalize_action):
    print('Validate Discriminator')
    batch_fake_states = []
    fake_action = []
    batch_real_states = expert_data[episode_idx:episode_idx +
                                    config.episodes_per_batch, 1:]  # frame 0 is condition
    real_action = expert_action[episode_idx:
                                episode_idx + config.episodes_per_batch, :-1]
    batch_real_states = np.concatenate(
        batch_real_states, axis=0)
    real_action = np.concatenate(real_action, axis=0)
    for _ in range(config.episodes_per_batch):
        # align the conditions with env
        # -1 : newest state
        conditions = expert_data[episode_idx:episode_idx+1, :, :]
        env.data = conditions[:, :, -1]
        obs_state = env.reset()
        for len_idx in range(config.max_length):
            if config.if_back_real:
                act = ppo_policy.act(
                    np.array(conditions[:, len_idx:len_idx+1, :]), stochastic=True)
            else:
                act = ppo_policy.act(
                    np.array(obs_state)[None, None], stochastic=True)
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
            fake_action.append(act.reshape([5, 2]))
    batch_fake_states = np.array(batch_fake_states)
    fake_action = np.array(fake_action)
    batch_real_states = normalize_observ(batch_real_states)
    # assert np.all(np.abs(batch_real_states[:, :, :6] - batch_fake_states[:, :, :6])<1e-2)
    assert batch_real_states.shape[0] == batch_fake_states.shape[0], "real: {}, fake: {}".format(
        batch_real_states.shape[0], batch_fake_states.shape[0])
    assert real_action.shape[0] == fake_action.shape[0], "real: {}, fake: {}".format(
        real_action.shape[0], fake_action.shape[0])
    D.validate(batch_fake_states, batch_real_states,
               fake_action, normalize_action(real_action))


def test_policy(config, vanilla_env, steps, ppo_policy, D, denormalize_observ):
    """ test policy
    - draw episode into mpeg video
    - collect episdoe and each state scores into numpy
    """
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
        numpy_collector.append(denormalize_observ(vanilla_obs))
    numpy_collector = np.array(numpy_collector)
    act_collector = np.array(act_collector)
    reward_collector = D.get_rewards_value(numpy_collector, act_collector)
    np.savez(os.path.join(config.logdir, 'gail_testing_{}/episode_{}.npz'.format(config.train_len, steps)),
             STATE=numpy_collector[:, -1], REWARD=reward_collector)


def tally_reward_line_chart(config, steps, ppo_policy, D, denormalize_observ, normalize_observ):
    """ tally 100 episodes as line chart to show how well the discriminator judge on each state of real and fake episode
    """
    episode_amount = 100
    # real data
    all_data = h5py.File(
        'bball_strategies/data/GAILTransitionData_51.hdf5', 'r')
    expert_data, _ = np.split(
        all_data['OBS'].value, [all_data['OBS'].value.shape[0]*9//10])
    expert_action, _ = np.split(
        all_data['DEF_ACT'].value, [all_data['DEF_ACT'].value.shape[0]*9//10])
    # env
    vanilla_env = gym.make(config.env)
    vanilla_env = BBallWrapper(vanilla_env, init_mode=1, fps=config.FPS, if_back_real=False,
                               time_limit=50)
    vanilla_env.data = np.load('bball_strategies/data/GAILEnvData_51.npy')
    # real
    selected_idx = np.random.choice(expert_data.shape[0], episode_amount)
    batch_real_states = expert_data[selected_idx, 1:] # frame 0 is condition
    real_action = expert_action[selected_idx, :-1]
    batch_real_states = np.concatenate(
        batch_real_states, axis=0)
    real_action = np.concatenate(real_action, axis=0)
    batch_real_states = normalize_observ(batch_real_states)
    real_rewards = D.get_rewards_value(
        batch_real_states, real_action).reshape([100, -1])
    # fake
    numpy_collector = []
    act_collector = []
    for _ in range(episode_amount):
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
            # numpy_collector.append(denormalize_observ(vanilla_obs))
            numpy_collector.append(vanilla_obs)
    numpy_collector = np.array(numpy_collector)
    act_collector = np.array(act_collector)
    fake_rewards = D.get_rewards_value(
        numpy_collector, act_collector).reshape([100, -1])
    # vis
    vis_line_chart(real_rewards, fake_rewards, config.logdir, str(steps))


def capped_video_schedule(episode_id):
    return episode_id % 10000 == 0


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
        min_ = dummy_env.observation_space.low
        max_ = dummy_env.observation_space.high
        observ = 2.0 * (observ - min_) / (max_ - min_) - 1.0
        return observ

    def normalize_action(act):
        min_ = dummy_env.action_space[3].low
        max_ = dummy_env.action_space[3].high
        act = 2.0 * (act - min_) / (max_ - min_) - 1.0
        return act

    def denormalize_observ(observ):
        min_ = dummy_env.observation_space.low
        max_ = dummy_env.observation_space.high
        observ = (observ + 1.0) * (max_ - min_) / 2.0 + min_
        return observ

    # env to testing
    vanilla_env = gym.make(config.env)
    vanilla_env = BBallWrapper(vanilla_env, init_mode=1, fps=config.FPS, if_back_real=False,
                               time_limit=50)
    vanilla_env = MonitorWrapper(vanilla_env, directory=os.path.join(config.logdir, 'gail_testing_{}/'.format(config.train_len)), if_back_real=False, video_callable=lambda _: True,
                                 # init from dataset
                                 init_mode=1)
    # if not os.path.exists(os.path.join(config.logdir, 'gail_testing')):
    #     os.makedirs(os.path.join(config.logdir, 'gail_testing'))
    vanilla_env.data = np.load('bball_strategies/data/GAILEnvData_51.npy')
    # env to generate fake state
    env = gym.make(config.env)
    env = BBallWrapper(env, init_mode=3, fps=config.FPS, if_back_real=config.if_back_real,
                       time_limit=config.max_length)
    env = MonitorWrapper(env, directory=os.path.join(config.logdir, 'gail_training/'), if_back_real=config.if_back_real,
                         # init from dataset in order
                         init_mode=3)
    # Discriminator graph
    with tf.device('/gpu:0'):
        D = Discriminator(config, dummy_env)
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
    ppo_policy = PPOPolicy(config, env)
    # Data
    all_data = h5py.File(
        'bball_strategies/data/GAILTransitionData_{}.hdf5'.format(config.train_len), 'r')
    expert_data, valid_expert_data = np.split(
        all_data['OBS'].value, [all_data['OBS'].value.shape[0]*9//10])
    expert_action, valid_expert_action = np.split(
        all_data['DEF_ACT'].value, [all_data['DEF_ACT'].value.shape[0]*9//10])
    print('expert_data', expert_data.shape)
    print('valid_expert_data', valid_expert_data.shape)
    print('expert_action', expert_action.shape)
    print('valid_expert_action', valid_expert_action.shape)

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
        # reset D optimizer
        D.reset_optimizer(sess)
        # reset PPO optimizer
        opt_reset = tf.group(
            [v.initializer for v in graph.algo._optimizer.variables()])
        sess.run(opt_reset)
        # visulization stuff
        if FLAGS.tally_only:
            tally_reward_line_chart(config, sess.run(D._global_steps), ppo_policy, D, denormalize_observ, normalize_observ)
            exit()
        # GAIL
        cumulate_steps = sess.run(graph.step)
        episode_idx = 0
        valid_episode_idx = 0
        while True:
            if episode_idx > (expert_data.shape[0]-config.episodes_per_batch*config.train_d_per_ppo) or episode_idx == 0:
                episode_idx = 0
                perm_idx = np.random.permutation(expert_data.shape[0])
                expert_data = expert_data[perm_idx]
                expert_action = expert_action[perm_idx]
            if valid_episode_idx > (valid_expert_data.shape[0]-config.episodes_per_batch) or valid_episode_idx == 0:
                valid_episode_idx = 0
                valid_perm_idx = np.random.permutation(
                    valid_expert_data.shape[0])
                valid_expert_data = valid_expert_data[valid_perm_idx]
                valid_expert_action = valid_expert_action[valid_perm_idx]
            # testing
            if valid_episode_idx % (100 * config.episodes_per_batch) == 0:
                test_policy(config, vanilla_env, sess.run(D._global_steps), ppo_policy,
                            D, denormalize_observ)
            if valid_episode_idx % (1000 * config.episodes_per_batch) == 0:
                tally_reward_line_chart(config, sess.run(D._global_steps), ppo_policy, D, denormalize_observ, normalize_observ)
            # train Discriminator
            train_Discriminator(
                episode_idx, config, expert_data, expert_action, env, ppo_policy, D, normalize_observ, normalize_action)
            # valid Discriminator
            valid_Discriminator(
                valid_episode_idx % (valid_expert_data.shape[0]-config.episodes_per_batch), config, valid_expert_data, valid_expert_action, env, ppo_policy, D, normalize_observ, normalize_action)
            episode_idx += config.episodes_per_batch*config.train_d_per_ppo
            valid_episode_idx += config.episodes_per_batch
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
    tf.app.flags.DEFINE_boolean(
        'tally_only', False,
        'whether to tally the reward line chart')
    tf.app.run()
