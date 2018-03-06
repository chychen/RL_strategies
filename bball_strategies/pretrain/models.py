"""
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from bball_strategies.gym_bball.envs.bball_env import BBallEnv


def denormalize_action(action):
    # skip discrete item (self._env.action_space[0])
    min_ = -BBallEnv().pl_max_power
    max_ = BBallEnv().pl_max_power
    action = (action + 1.0) / 2.0 * (max_ - min_) + min_
    return action

class PretrainOffense(object):
    """ supervised pretraining for offensive policy
    """

    def __init__(self, config):
        self._config = config
        self._steps = tf.train.get_or_create_global_step()
        self._input_obs = tf.placeholder(
            tf.float32, shape=[None, None, 5, 14, 2], name='input_obs')
        self._label = tf.placeholder(
            tf.float32, shape=[None, None, 15], name='label')
        self._label_decision = self._label[:,:,:3]
        self._label_action = self._label[:,:,3:]
        # inference
        self._logits, self._off_action_mean = self._config.network(
            self._config, self._input_obs)
        self._off_action_mean = denormalize_action(self._off_action_mean)
        self._loss, self._summary = self.loss_function()
        # train
        optimizer = self._config.optimizer(self._config.learning_rate)
        self._train_op = optimizer.minimize(
            self._loss, global_step=self._steps)
        # TODO summary trainable variables

    def loss_function(self):
        """
        """
        # decision
        decision_loss = tf.losses.softmax_cross_entropy(
            self._label_decision[:, 0], self._logits[:, 0])
        # action
        action_loss = tf.losses.mean_squared_error(
            self._label_action, self._off_action_mean)
        # weighted sum
        with tf.control_dependencies([tf.assert_less(self._config.loss_alpha, 1.0)]):
            loss = self._config.loss_alpha * decision_loss + \
                (1.0-self._config.loss_alpha) * action_loss
        # summary
        summary = tf.summary.merge([
            tf.summary.scalar('decision_loss', decision_loss),
            tf.summary.scalar('action_loss', action_loss),
            tf.summary.scalar('loss', loss)
        ])
        return loss, summary

    def train(self, sess, data, label):
        feed_dict = {
            self._input_obs: data,
            self._label: label
        }
        summary, loss, steps, _ = sess.run(
            [self._summary, self._loss, self._steps, self._train_op], feed_dict=feed_dict)
        return summary, loss, steps

    def eval(self, sess, data, label):
        feed_dict = {
            self._input_obs: data,
            self._label: label
        }
        summary, loss, steps = sess.run(
            [self._summary, self._loss, self._steps], feed_dict=feed_dict)
        return summary, loss, steps

    def perform(self, sess, data):
        feed_dict = {
            self._input_obs: data
        }
        logits, actions = sess.run(
            [self._logits, self._off_action_mean], feed_dict=feed_dict)
        return logits, actions


class PretrainDefense(object):
    """ supervised pretraining for defensice policy
    """

    def __init__(self, config):
        self._config = config
        self._steps = tf.train.get_or_create_global_step()
        self._input_obs = tf.placeholder(
            tf.float32, shape=[None, None, 5, 14, 2], name='input_obs')
        self._label = tf.placeholder(
            tf.float32, shape=[None, None, 10], name='label')
        # inference
        self._def_action_mean = self._config.network(
            self._config, self._input_obs)
        self._def_action_mean = denormalize_action(self._def_action_mean)
        self._loss, self._summary = self.loss_function()
        # train
        optimizer = self._config.optimizer(self._config.learning_rate)
        self._train_op = optimizer.minimize(
            self._loss, global_step=self._steps)
        # TODO summary trainable variables

    def loss_function(self):
        """
        """
        # action
        loss = tf.losses.mean_squared_error(
            self._label, self._def_action_mean)
        summary = tf.summary.scalar('loss', loss)
        return loss, summary

    def train(self, sess, data, label):
        feed_dict = {
            self._input_obs: data,
            self._label: label
        }
        summary, loss, steps, _ = sess.run(
            [self._summary, self._loss, self._steps, self._train_op], feed_dict=feed_dict)
        return summary, loss, steps

    def eval(self, sess, data, label):
        feed_dict = {
            self._input_obs: data,
            self._label: label
        }
        summary, loss, steps = sess.run(
            [self._summary, self._loss, self._steps], feed_dict=feed_dict)
        return summary, loss, steps

    def perform(self, sess, data):
        feed_dict = {
            self._input_obs: data
        }
        actions = sess.run(
            self._def_action_mean, feed_dict=feed_dict)
        return actions


def main():
    # test model runable
    pass


if __name__ == '__main__':
    main()
