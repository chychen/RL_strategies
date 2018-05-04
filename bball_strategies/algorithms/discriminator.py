""" Model of Critic Network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import numpy as np
import tensorflow as tf


def get_var_list(prefix):
    """ to get both Generator's trainable variables and add trainable variables into histogram summary

    Params
    ------
    prefix : string
        string to select out the trainalbe variables
    """
    trainable_V = tf.trainable_variables()
    theta = []
    for _, v in enumerate(trainable_V):
        if v.name.startswith(prefix):
            theta.append(v)
    return theta


class Discriminator(object):
    __instance = None

    def __new__(cls, config=None, env=None):
        if Discriminator.__instance is None:
            Discriminator.__instance = object.__new__(cls)
        else:
            print("Instance Exists! :D")
        return Discriminator.__instance

    def __init__(self, config=None, env=None):
        """
        state is composed of offense(condition) and defense
        """
        if self.__instance is not None and config is not None:
            # first init the class
            self._global_steps = tf.train.get_or_create_global_step()
            with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
                self.__steps = tf.get_variable('D_steps', shape=[
                ], dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32), trainable=False)
                # state shape = [batch_size, buffer_size, 14, 2]
                self._expert_s = tf.placeholder(dtype=tf.float32, shape=[
                                                None]+list(env.observation_space.shape))
                self._agent_s = tf.placeholder(dtype=tf.float32, shape=[
                                               None]+list(env.observation_space.shape))
                self._batch_size = tf.shape(self._expert_s)[0]
                self._buffer_size = list(env.observation_space.shape)[0]
                self._config = config
                self._loss = self.__loss_function()
                with tf.name_scope('optimizer'):
                    theta = get_var_list('Discriminator')
                    # Discriminator train one iteration, step++
                    assign_add_ = tf.assign_add(self.__steps, 1)
                    with tf.control_dependencies([assign_add_]):
                        optimizer = self._config.optimizer(
                            learning_rate=self._config.learning_rate)  # TODO beta1=0.5, beta2=0.9
                        grads = tf.gradients(self._loss, theta)
                        grads = list(zip(grads, theta))
                        self._train_op = optimizer.apply_gradients(
                            grads_and_vars=grads, global_step=self._global_steps)
            # summary
            log_path = os.path.join(self._config.logdir, 'Discriminator')
            self._summary_op = tf.summary.merge(tf.get_collection('D'))
            self._summary_valid_op = tf.summary.merge(
                tf.get_collection('D_valid'))
            self.summary_writer = tf.summary.FileWriter(
                log_path + '/D')
            self.valid_summary_writer = tf.summary.FileWriter(
                log_path + '/D_valid')

    def __loss_function(self):
        with tf.name_scope('wgan_gp_loss'):
            # TODO improve of improved WGAN
            epsilon = tf.random_uniform(
                [self._batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            # only defense part is different, others are conditions
            # conditions part might be canceled out to 0
            X_inter = epsilon * self._expert_s + (1.0-epsilon) * self._agent_s
            # add back the conditions
            X_inter = tf.concat(
                [self._expert_s[:, :, 0:6], X_inter[:, :, 6:11]], axis=2)
            # if transition-based
            X_inter = tf.concat(
                [self._expert_s[:, :self._buffer_size-1, :], X_inter[:, -1:, :]], axis=1)
            grad = tf.gradients(self._config.d_network(
                X_inter, reuse=False), [X_inter])[0]
            sum_ = tf.reduce_sum(tf.square(grad), axis=0)
            grad_norm = tf.sqrt(sum_)
            grad_pen = self._config.wgan_penalty_lambda * tf.reduce_mean(
                tf.square(grad_norm - 1.0))
            fake_scores = self._config.d_network(self._agent_s, reuse=True)
            real_scores = self._config.d_network(self._expert_s, reuse=True)
            f_fake = tf.reduce_mean(fake_scores)
            f_real = tf.reduce_mean(real_scores)
            em_distance = f_real - f_fake
            loss = -em_distance + grad_pen

            # logging
            tf.summary.scalar('D_loss', loss,
                              collections=['D', 'D_valid'])
            tf.summary.scalar('F_real', f_real, collections=['D'])
            tf.summary.scalar('F_fake', f_fake, collections=['D'])
            tf.summary.scalar('Earth Moving Distance',
                              em_distance, collections=['D', 'D_valid'])
            tf.summary.scalar('grad_pen', grad_pen, collections=['D'])

            return loss

    def train(self, agent_s, expert_s):
        feed_dict = {
            self._expert_s: expert_s,
            self._agent_s: agent_s
        }
        global_step, _, summary = tf.get_default_session().run(
            [self._global_steps, self._train_op, self._summary_op], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step=global_step)
        return

    def validate(self, agent_s, expert_s):
        feed_dict = {
            self._expert_s: expert_s,
            self._agent_s: agent_s
        }
        global_step, summary = tf.get_default_session().run(
            [self._global_steps, self._summary_valid_op], feed_dict=feed_dict)
        self.valid_summary_writer.add_summary(summary, global_step=global_step)
        pass

    def get_rewards(self, state):
        with tf.variable_scope('Discriminator'):
            return self._config.d_network(state, reuse=True)


def main():
    """ test
    """
    from bball_strategies.scripts.gail import configs
    from agents import tools
    import gym
    config = tools.AttrDict(configs.default())
    with config.unlocked:
        config.logdir = 'test'
    env = gym.make(config.env)
    D1 = Discriminator(config, env)
    D2 = Discriminator()
    dummy = tf.ones(shape=[128, 10, 14, 2])
    r = D2.get_rewards(dummy)
    print(r)


if __name__ == '__main__':
    main()
