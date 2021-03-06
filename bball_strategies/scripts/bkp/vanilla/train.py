
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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import gym
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from agents import tools
from agents.scripts import utility
from bball_strategies import gym_bball
from bball_strategies.scripts.vanilla import configs
from bball_strategies.gym_bball.tools import BBallWrapper


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
    # loop.add_phase(
    #     'train_defense', graph.done, graph.score, graph.summary, train_steps,
    #     report_every=train_steps,
    #     log_every=train_steps,
    #     checkpoint_every=None,
    #     feed={graph.is_training: True,
    #           graph.is_optimizing_offense: False})
    # loop.add_phase(
    #     'eval_offense', graph.done, graph.score, graph.summary, eval_steps,
    #     report_every=eval_steps,
    #     log_every=eval_steps,
    #     checkpoint_every=10 * eval_steps,
    #     feed={graph.is_training: False,
    #           graph.is_optimizing_offense: True})
    # loop.add_phase(
    #     'eval_defense', graph.done, graph.score, graph.summary, eval_steps,
    #     report_every=eval_steps,
    #     log_every=eval_steps,
    #     checkpoint_every=10 * eval_steps,
    #     feed={graph.is_training: False,
    #           graph.is_optimizing_offense: False})
    return loop


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
    if FLAGS.off_ckpt and FLAGS.def_ckpt:
        # restore both offense and defense pretrain model
        off_saver = utility.define_saver_with_prefix(
            exclude=(r'.*d_trunk/.*', r'.*value/.*', r'.*two_trunk_gaussian/.*'))
        def_saver = utility.define_saver_with_prefix(
            exclude=(r'.*o_trunk/.*', r'.*value/.*', r'.*two_trunk_gaussian/.*'))
        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=config.log_device_placement)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(
                    sess, ui_type=FLAGS.ui_type)
            utility.initialize_pretrained_variables(
                sess, off_saver, FLAGS.off_ckpt, def_saver, FLAGS.def_ckpt)
            for score in loop.run(sess, saver, total_steps):
                yield score

    else:
        sess_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=config.log_device_placement)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(
                    sess, ui_type=FLAGS.ui_type)
            utility.initialize_variables(
                sess, saver, config.logdir, resume=FLAGS.resume)
            for score in loop.run(sess, saver, total_steps):
                yield score
    batch_env.close()


def main(_):
    """ Create or load configuration and launch the trainer.
    """
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

    for score in train(config, FLAGS.env_processes, outdir):
        tf.logging.info('Score {}.'.format(score))


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', 'logdir/offense_only',
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
        'debug', False,
        'whether to enable debug mode')
    tf.app.flags.DEFINE_boolean(
        'resume', False,
        'whether to restore checkpoint in logdir')
    tf.app.flags.DEFINE_string(
        'off_ckpt', None,
        'Specify what checkpoint name to use for offense')
    tf.app.flags.DEFINE_string(
        'def_ckpt', None,
        'Specify what checkpoint name to use for defense')
    tf.app.flags.DEFINE_string(
        'ui_type', 'curses',
        "Command-line user interface type (curses | readline)")
    tf.app.run()
