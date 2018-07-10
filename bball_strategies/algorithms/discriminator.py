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
import gym
from bball_strategies import gym_bball


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

    def __new__(cls, agent_s=None, agent_a=None, expert_s=None, expert_a=None, config=None):
        if Discriminator.__instance is None:
            Discriminator.__instance = object.__new__(cls)
        else:
            print("Instance Exists! :D")
        return Discriminator.__instance

    def __init__(self, agent_s=None, agent_a=None, expert_s=None, expert_a=None, config=None):
        """
        state is composed of offense(condition) and defense
        """
        if self.__instance is not None and config is not None:
            env = gym.make(config.env)
            # first init the class
            # self._global_steps = tf.train.get_or_create_global_step()
            with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
                self._steps = tf.get_variable('D_steps', shape=[
                ], dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32), trainable=False)
                # state shape = [batch_size, buffer_size, 14, 2]
                # self._expert_s = tf.placeholder(dtype=tf.float32, shape=[
                #                                 None, None] + list(env.observation_space.shape[1:]))
                self._expert_s = expert_s
                self._agent_s_ph = tf.placeholder(dtype=tf.float32, shape=[
                    None, None] + list(env.observation_space.shape[1:]))
                self._agent_s = agent_s
                # self._expert_a = tf.placeholder(dtype=tf.float32, shape=[
                #                                 None, None, 5, 2])
                self._expert_a = expert_a
                self._agent_a_ph = tf.placeholder(dtype=tf.float32, shape=[
                    None, None, 5, 2])
                self._agent_a = agent_a
                self._batch_size = tf.shape(self._expert_s)[0]
                self._buffer_size = tf.shape(self._expert_s)[1]
                self._config = config
                self._loss, self.f_fake, self.f_real, self.em_distance, self.grad_pen = self.__loss_function()
                with tf.name_scope('optimizer'):
                    theta = get_var_list('Discriminator')
                    # Discriminator train one iteration, step++
                    # assign_add_ = tf.assign_add(self._steps, 1)
                    # with tf.control_dependencies([assign_add_]):
                    self.optimizer = self._config.optimizer(
                        learning_rate=self._config.learning_rate)  # TODO beta1=0.5, beta2=0.9 recommended by WGAN-GP
                    grads = tf.gradients(self._loss, theta)
                    grads = list(zip(grads, theta))
                    self._train_op = self.optimizer.apply_gradients(
                        grads_and_vars=grads, global_step=self._steps)

    def __loss_function(self):
        with tf.name_scope('wgan_gp_loss'):
            # TODO improve of improved WGAN
            epsilon = tf.random_uniform(
                [self._batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            # only defense part is different, others are conditions
            # conditions part might be canceled out to 0
            X_inter = epsilon * self._expert_s + \
                (1.0 - epsilon) * self._agent_s
            X_act_inter = epsilon * self._expert_a + \
                (1.0 - epsilon) * self._agent_a
            # add back the conditions
            X_inter = tf.concat(
                [self._expert_s[:, :, 0:6], X_inter[:, :, 6:11], self._expert_s[:, :, 11:14]], axis=2)
            d_out_inter, _ = self._config.d_network(
                X_inter, X_act_inter, reuse=tf.AUTO_REUSE)
            grad_obs, grad_act = tf.gradients(d_out_inter, [X_inter, X_act_inter])
            grad_obs = tf.reshape(grad_obs, shape=[self._batch_size, -1])
            grad_act = tf.reshape(grad_act, shape=[self._batch_size, -1])
            grad = tf.concat([grad_obs, grad_act], axis=1)

            sum_ = tf.reduce_sum(tf.square(grad), axis=0)
            grad_norm = tf.sqrt(sum_)
            grad_pen = self._config.wgan_penalty_lambda * tf.reduce_mean(
                tf.square(grad_norm - 1.0))
            fake_scores, _ = self._config.d_network(
                self._agent_s, self._agent_a, reuse=tf.AUTO_REUSE)
            real_scores, _ = self._config.d_network(
                self._expert_s, self._expert_a, reuse=tf.AUTO_REUSE)
            f_fake = tf.reduce_mean(fake_scores)
            f_real = tf.reduce_mean(real_scores)
            em_distance = f_real - f_fake
            loss = -em_distance + grad_pen

            return loss, f_fake, f_real, em_distance, grad_pen

    @classmethod
    def get_rewards(cls, state, action, config):
        with tf.variable_scope('Discriminator'):
            rewards, _ = config.d_network(state, action, reuse=tf.AUTO_REUSE)
            return rewards

    def get_rewards_value(self, state, action):
        feed_dict = {
            self._agent_s_ph: state,
            self._agent_a_ph: action
        }
        with tf.variable_scope('Discriminator'):
            _, fake_scores_by_frame = self._config.d_network(
                self._agent_s_ph, self._agent_a_ph, reuse=tf.AUTO_REUSE)
            return tf.get_default_session().run(fake_scores_by_frame, feed_dict=feed_dict)


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
