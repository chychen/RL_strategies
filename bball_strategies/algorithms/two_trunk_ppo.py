"""Proximal Policy Optimization agent.

Based on
- John Schulman's implementation in Python and Theano:
https://github.com/joschu/modular_rl/blob/master/modular_rl/ppo.py
- Tensorflow Agents:
https://github.com/tensorflow/agents/blob/master/agents/algorithms/ppo/ppo.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from agents import parts
from agents import tools
from agents.algorithms.ppo import utility


class TWO_TRUNK_PPO(object):
    """A vectorized implementation of the PPO algorithm by John Schulman."""

    def __init__(self, batch_env, step, is_training, should_log, config, is_optimizing_offense):
        """Create an instance of the PPO algorithm.

        Args:
        -----
        batch_env : In-graph batch environment.
        step : Integer tensor holding the current training step.
        is_training : Boolean tensor for whether the algorithm should train.
        should_log : Boolean tensor for whether summaries should be returned.
        config : Object containing the agent configuration as attributes.
        is_optimizing_offense : (Extended) whose turn to optimize
        """
        self._batch_env = batch_env
        self._step = step
        self._is_training = is_training
        self._is_optimizing_offense = is_optimizing_offense
        self._should_log = should_log
        self._config = config
        self._observ_filter = parts.StreamingNormalize(
            self._batch_env.observ[0], center=True, scale=True, clip=5,
            name='normalize_observ')
        self._reward_filter = parts.StreamingNormalize(
            self._batch_env.reward[0], center=False, scale=True, clip=10,
            name='normalize_reward')
        self._use_gpu = self._config.use_gpu and utility.available_gpus()
        policy_params, state = self._initialize_policy()
        self._initialize_memory(policy_params)
        # Initialize the optimizer and penalty.
        with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
            self._optimizer = self._config.optimizer(
                self._config.learning_rate)
        self._penalty = tf.Variable(
            self._config.kl_init_penalty, False, dtype=tf.float32)
        # If the policy is stateful, allocate space to store its state.
        with tf.variable_scope('ppo_temporary'):
            with tf.device('/gpu:0'):
                if state is None:
                    self._last_state = None
                else:
                    def var_like(x): return tf.Variable(
                        lambda: tf.zeros_like(x), False)
                    self._last_state = tools.nested.map(var_like, state)
        # Remember the action and policy parameters to write into the memory.
        with tf.variable_scope('ppo_temporary'):
            self._last_action = tf.Variable(
                tf.zeros_like(self._batch_env.action), False, name='last_action')
            self._last_policy = tools.nested.map(
                lambda x: tf.Variable(tf.zeros_like(x[:, 0], optimize=False)), policy_params)

    def begin_episode(self, agent_indices):
        """Reset the recurrent states and stored episode.

        Args:
          agent_indices: Tensor containing current batch indices.

        Returns:
          Summary tensor.
        """
        with tf.name_scope('begin_episode/'):
            if self._last_state is None:
                reset_state = tf.no_op()
            else:
                reset_state = utility.reinit_nested_vars(
                    self._last_state, agent_indices)
            reset_buffer = self._current_episodes.clear(agent_indices)
            with tf.control_dependencies([reset_state, reset_buffer]):
                return tf.constant('')

    def perform(self, agent_indices, observ, turn_info):
        """Compute batch of actions and a summary for a batch of observation.

        Args:
          agent_indices: Tensor containing current batch indices.
          observ: Tensor of a batch of observations for all agents.

        Returns:
          Tuple of action batch tensor and summary tensor.
        """
        with tf.name_scope('perform/'):
            observ = self._observ_filter.transform(observ)
            if self._last_state is None:
                state = None
            else:
                state = tools.nested.map(
                    lambda x: tf.gather(x, agent_indices), self._last_state)
            with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
                output = self._network(
                    observ[:, None], tf.ones(observ.shape[0]), state)
            # offense turn sample
            off_turn_sample = tf.concat([
                tf.cast(output.policy[ACT['DECISION']].sample()[
                        :, :, None], tf.float32),
                output.policy[ACT['OFF_DASH']].sample(),
                tf.zeros(shape=[observ.shape[0], 1, 10])], axis=2)
            off_turn_mode = tf.concat([
                tf.cast(output.policy[ACT['DECISION']].mode()[
                        :, :, None], tf.float32),
                output.policy[ACT['OFF_DASH']].mode(),
                tf.zeros(shape=[observ.shape[0], 1, 10])], axis=2)
            # defense turn policy
            def_turn_sample = tf.concat([
                tf.zeros(shape=[observ.shape[0], 1, 1]),
                tf.zeros(shape=[observ.shape[0], 1, 11]),
                output.policy[ACT['DEF_DASH']].sample()], axis=2)
            def_turn_mode = tf.concat([
                tf.zeros(shape=[observ.shape[0], 1, 1]),
                tf.zeros(shape=[observ.shape[0], 1, 11]),
                output.policy[ACT['DEF_DASH']].mode()], axis=2)
            sample_ = tf.where(tf.equal(turn_info, 0),
                               off_turn_sample, def_turn_sample)
            mode_ = tf.where(tf.equal(turn_info, 0),
                             off_turn_mode, def_turn_mode)
            action = tf.where(
                self._is_training, sample_, mode_)
            action = tf.reshape(action, shape=[observ.shape[0], 11, 2])

            # TODO summary
            summary = str()
            # logprob = output.policy.log_prob(action)[:, 0]
            # # pylint: disable=g-long-lambda
            # summary = tf.cond(self._should_log, lambda: tf.summary.merge([
            #     tf.summary.histogram('mode', mode_),
            #     tf.summary.histogram('action', action_reshape),
            #     tf.summary.histogram('logprob', logprob)]), str)

            # Remember current policy to append to memory in the experience callback.
            if self._last_state is None:
                assign_state = tf.no_op()
            else:
                assign_state = utility.assign_nested_vars(
                    self._last_state, output.state, agent_indices)
            remember_last_action = tf.scatter_update(
                self._last_action, agent_indices, action)

            def is_tensor(x):
                return isinstance(x, tf.Tensor)

            def set_batch_dim(x):
                return utility.set_dimension(
                    x, 0, len(self._batch_env))

            policy_params = []
            for _, v in ACT.items():
                policy_params.append(tools.nested.filter(
                    is_tensor, output.policy[v].parameters))
                tools.nested.map(set_batch_dim, policy_params[v])

            assert policy_params, 'Policy has no parameters to store.'
            remember_last_policy = tools.nested.map(
                lambda var, val: tf.scatter_update(
                    var, agent_indices, val[:, 0]),
                self._last_policy, policy_params, flatten=True)
            with tf.control_dependencies((
                    assign_state, remember_last_action) + remember_last_policy):
                return action, tf.identity(summary)

    def experience(
            self, agent_indices, observ, action, reward, unused_done, unused_nextob, turn_info):
        """Process the transition tuple of the current step.

        When training, add the current transition tuple to the memory and update
        the streaming statistics for observations and rewards. A summary string is
        returned if requested at this step.

        Args
        ----
          agent_indices: Tensor containing current batch indices.
          observ: Batch tensor of observations.
          action: Batch tensor of actions.
          reward: Batch tensor of rewards.
          unused_done: Batch tensor of done flags.
          unused_nextob: Batch tensor of successor observations.
          turn_info: Batch tensor of whose turn is it.

        Returns:
          Summary tensor.
        """
        with tf.name_scope('experience/'):
            return tf.cond(
                self._is_training,
                # pylint: disable=g-long-lambda
                lambda: self._define_experience(
                    agent_indices, observ, action, reward, turn_info=turn_info), str)

    def _define_experience(self, agent_indices, observ, action, reward, turn_info):
        """Implement the branch of experience() entered during training."""
        update_filters = tf.summary.merge([
            self._observ_filter.update(observ),
            self._reward_filter.update(reward)])
        with tf.control_dependencies([update_filters]):
            if self._config.train_on_agent_action:
                # NOTE: Doesn't seem to change much.
                action = self._last_action
            policy = tools.nested.map(
                lambda x: tf.gather(x, agent_indices), self._last_policy)
            batch = (observ, action, policy, reward)
            append = self._current_episodes.append(batch, agent_indices)
        with tf.control_dependencies([append]):
            norm_observ = self._observ_filter.transform(observ)
            norm_reward = tf.reduce_mean(self._reward_filter.transform(reward))
            # pylint: disable=g-long-lambda
            summary = tf.cond(self._should_log, lambda: tf.summary.merge([
                update_filters,
                self._observ_filter.summary(),
                self._reward_filter.summary(),
                tf.summary.scalar('memory_size', self._num_finished_episodes),
                tf.summary.histogram('normalized_observ', norm_observ),
                tf.summary.histogram('action', self._last_action),
                tf.summary.scalar('normalized_reward', norm_reward)]), str)
            return summary

    def end_episode(self, agent_indices):
        """Add episodes to the memory and perform update steps if memory is full.

        During training, add the collected episodes of the batch indices that
        finished their episode to the memory. If the memory is full, train on it,
        and then clear the memory. A summary string is returned if requested at
        this step.

        Args:
          agent_indices: Tensor containing current batch indices.

        Returns:
           Summary tensor.
        """
        with tf.name_scope('end_episode/'):
            return tf.cond(
                self._is_training,
                lambda: self._define_end_episode(agent_indices), str)

    def _initialize_policy(self):
        """ #### Initialize the policy.

        Run the policy network on dummy data to initialize its parameters for later
        reuse and to analyze the policy distribution. Initializes the attributes
        `self._network` and `self._policy_type`.

        Raises
        ------
          ValueError: Invalid policy distribution.

        Returns
        -------
          Parameters of the policy distribution and policy state for both offense and deffense.
        """
        with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
            network = functools.partial(
                self._config.network, self._config, self._batch_env.action_space)
            self._network = tf.make_template('network', network)
            output = self._network(
                tf.zeros_like(self._batch_env.observ[:, None]),
                tf.ones(len(self._batch_env)))
        # shape checking
        decision_shape = output.policy[ACT['DECISION']].event_shape
        off_dash_shape = output.policy[ACT['OFF_DASH']].event_shape
        def_dash_shape = output.policy[ACT['DEF_DASH']].event_shape
        if decision_shape != () or off_dash_shape != (11,) or def_dash_shape != (10,):
            message = 'Policy event shape does not match action shape.'
            raise ValueError(message)
        self._policy_type = [type(output.policy[ACT['DECISION']]),
                             type(output.policy[ACT['OFF_DASH']]),
                             type(output.policy[ACT['DEF_DASH']])]

        def is_tensor(x):
            return isinstance(x, tf.Tensor)

        def set_batch_dim(x):
            return utility.set_dimension(
                x, 0, len(self._batch_env))

        policy_params = []
        for _, v in ACT.items():
            policy_params.append(tools.nested.filter(
                is_tensor, output.policy[v].parameters))
            tools.nested.map(set_batch_dim, policy_params[v])

        if output.state is not None:
            tools.nested.map(set_batch_dim, output.state)
        return policy_params, output.state

    def _initialize_memory(self, policy_params):
        """Initialize temporary and permanent memory.

        Args:
          policy_params: Nested tuple of policy parametres with all dimensions set.
          policy_params[0]: offense dicision
          policy_params[1]: offense dash
          policy_params[2]: defense dash

        Initializes the attributes `self._current_episodes`,
        `self._finished_episodes`, and `self._num_finished_episodes`. The episodes
        memory serves to collect multiple episodes in parallel. Finished episodes
        are copied into the next free slot of the second memory. The memory index
        points to the next free slot.
        """
        # We store observation, action, policy parameters, and reward.
        template = (
            self._batch_env.observ[0],
            self._batch_env.action[0],
            tools.nested.map(lambda x: x[0, 0], policy_params),
            self._batch_env.reward[0])
        with tf.variable_scope('ppo_temporary'):
            self._current_episodes = parts.EpisodeMemory(
                template, len(self._batch_env), self._config.max_length, 'episodes')
        self._finished_episodes = parts.EpisodeMemory(
            template, self._config.update_every, self._config.max_length, 'memory')
        self._num_finished_episodes = tf.Variable(0, False)

    def _define_end_episode(self, agent_indices):
        """Implement the branch of end_episode() entered during training."""
        episodes, length = self._current_episodes.data(agent_indices)
        space_left = self._config.update_every - self._num_finished_episodes
        use_episodes = tf.range(tf.minimum(
            tf.shape(agent_indices)[0], space_left))
        episodes = tools.nested.map(
            lambda x: tf.gather(x, use_episodes), episodes)
        append = self._finished_episodes.replace(
            episodes, tf.gather(length, use_episodes),
            use_episodes + self._num_finished_episodes)
        with tf.control_dependencies([append]):
            increment_index = self._num_finished_episodes.assign_add(
                tf.shape(use_episodes)[0])
        with tf.control_dependencies([increment_index]):
            memory_full = self._num_finished_episodes >= self._config.update_every
            return tf.cond(memory_full, self._training, str)

    def _training(self):
        """Perform multiple training iterations of both policy and value baseline.

        Training on the episodes collected in the memory. Reset the memory
        afterwards. Always returns a summary string.

        Returns:
          Summary tensor.
        """
        with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
            with tf.name_scope('training'):
                assert_full = tf.assert_equal(
                    self._num_finished_episodes, self._config.update_every)
                with tf.control_dependencies([assert_full]):
                    data = self._finished_episodes.data()
                (observ, action, old_policy_params, reward), length = data
                # We set padding frames of the parameters to ones to prevent Gaussians
                # with zero variance. This would result in an inifite KL divergence,
                # which, even if masked out, would result in NaN gradients.
                old_policy_params = tools.nested.map(
                    lambda param: self._mask(param, length, 1), old_policy_params)
                with tf.control_dependencies([tf.assert_greater(length, 0)]):
                    length = tf.identity(length)
                observ = self._observ_filter.transform(observ)
                reward = self._reward_filter.transform(reward)
                update_summary = self._perform_update_steps(
                    observ, action, old_policy_params, reward, length)
                with tf.control_dependencies([update_summary]):
                    penalty_summary = self._adjust_penalty(
                        observ, old_policy_params, length)
                with tf.control_dependencies([penalty_summary]):
                    clear_memory = tf.group(
                        self._finished_episodes.clear(),
                        self._num_finished_episodes.assign(0))
                with tf.control_dependencies([clear_memory]):
                    weight_summary = utility.variable_summaries(
                        tf.trainable_variables(), self._config.weight_summaries)
                    return tf.summary.merge([
                        update_summary, penalty_summary, weight_summary])

    def _perform_update_steps(
            self, observ, action, old_policy_params, reward, length):
        """Perform multiple update steps of value function and policy.

        The advantage is computed once at the beginning and shared across
        iterations. We need to decide for the summary of one iteration, and thus
        choose the one after half of the iterations.

        Args:
          observ: Sequences of observations.
          action: Sequences of actions.
          old_policy_params: Parameters of the behavioral policy.
          reward: Sequences of rewards.
          length: Batch of sequence lengths.

        Returns:
          Summary tensor.
        """
        return_ = utility.discounted_return(
            reward, length, self._config.discount)
        value = self._network(observ, length).value
        value = tf.where(self._is_optimizing_offense,
                         value[TEAM['OFFENSE']], value[TEAM['DEFENSE']])
        if self._config.gae_lambda:
            advantage = utility.lambda_advantage(
                reward, value, length, self._config.discount,
                self._config.gae_lambda)
        else:
            advantage = return_ - value
        mean, variance = tf.nn.moments(advantage, axes=[0, 1], keep_dims=True)
        advantage = (advantage - mean) / (tf.sqrt(variance) + 1e-8)
        advantage = tf.Print(
            advantage, [tf.reduce_mean(return_), tf.reduce_mean(value)],
            'return and value: ')
        advantage = tf.Print(
            advantage, [tf.reduce_mean(advantage)],
            'normalized advantage: ')
        episodes = (observ, action,
                    old_policy_params[ACT['DECISION']],
                    old_policy_params[ACT['OFF_DASH']],
                    old_policy_params[ACT['DEF_DASH']],
                    reward, advantage)
        value_loss, policy_loss, summary = parts.iterate_sequences(
            self._update_step, [0., 0., ''], episodes, length,
            self._config.chunk_length,
            self._config.batch_size,
            self._config.update_epochs,
            padding_value=1)
        print_losses = tf.group(
            tf.Print(0, [tf.reduce_mean(value_loss)], 'value loss: '),
            tf.Print(0, [tf.reduce_mean(policy_loss)], 'policy loss: '))
        with tf.control_dependencies([value_loss, policy_loss, print_losses]):
            return summary[self._config.update_epochs // 2]

    def _update_step(self, sequence):
        """Compute the current combined loss and perform a gradient update step.

        The sequences must be a dict containing the keys `length` and `sequence`,
        where the latter is a tuple containing observations, actions, parameters of
        the behavioral policy, rewards, and advantages.

        Args:
          sequence: Sequences of episodes or chunks of episodes.

        Returns:
          Tuple of value loss, policy loss, and summary tensor.
        """
        observ, action, DECISION, OFF_DASH, DEF_DASH, reward, advantage = sequence['sequence']
        length = sequence['length']
        old_policy = []
        old_policy.append(self._policy_type[ACT['DECISION']](**DECISION))
        old_policy.append(self._policy_type[ACT['OFF_DASH']](**OFF_DASH))
        old_policy.append(self._policy_type[ACT['DEF_DASH']](**DEF_DASH))
        value_loss, value_summary = self._value_loss(observ, reward, length)
        network = self._network(observ, length)
        policy_loss, policy_summary = self._policy_loss(
            old_policy, network.policy, action, advantage, length)
        value_gradients, value_variables = (
            zip(*self._optimizer.compute_gradients(value_loss)))
        policy_gradients, policy_variables = (
            zip(*self._optimizer.compute_gradients(policy_loss)))
        all_gradients = value_gradients + policy_gradients
        all_variables = value_variables + policy_variables
        optimize = self._optimizer.apply_gradients(
            zip(all_gradients, all_variables))
        summary = tf.summary.merge([
            value_summary, policy_summary,
            tf.summary.scalar(
                'value_gradient_norm', tf.global_norm(value_gradients)),
            tf.summary.scalar(
                'policy_gradient_norm', tf.global_norm(policy_gradients)),
            utility.gradient_summaries(
                zip(value_gradients, value_variables), dict(value=r'.*')),
            utility.gradient_summaries(
                zip(policy_gradients, policy_variables), dict(policy=r'.*'))])
        with tf.control_dependencies([optimize]):
            return [tf.identity(x) for x in (value_loss, policy_loss, summary)]

    def _value_loss(self, observ, reward, length):
        """Compute the loss function for the value baseline.

        The value loss is the difference between empirical and approximated returns
        over the collected episodes. Returns the loss tensor and a summary strin.

        Args:
          observ: Sequences of observations.
          reward: Sequences of reward.
          length: Batch of sequence lengths.

        Returns:
          Tuple of loss tensor and summary tensor.
        """
        with tf.name_scope('value_loss'):
            value = self._network(observ, length).value
            value = tf.where(self._is_optimizing_offense,
                             value[TEAM['OFFENSE']], value[TEAM['DEFENSE']])
            return_ = utility.discounted_return(
                reward, length, self._config.discount)
            advantage = return_ - value
            value_loss = 0.5 * self._mask(advantage ** 2, length)
            summary = tf.summary.merge([
                tf.summary.histogram('value_loss', value_loss),
                tf.summary.scalar('avg_value_loss', tf.reduce_mean(value_loss))])
            value_loss = tf.reduce_mean(value_loss)
            return tf.check_numerics(value_loss, 'value_loss'), summary

    def _policy_loss(
            self, old_policy, policy, action, advantage, length):
        """Compute the policy loss composed of multiple components.

        1. The policy gradient loss is importance sampled from the data-collecting
           policy at the beginning of training.
        2. The second term is a KL penalty between the policy at the beginning of
           training and the current policy.
        3. Additionally, if this KL already changed more than twice the target
           amount, we activate a strong penalty discouraging further divergence.

        Args:
          old_policy: Action distribution of the behavioral policy.
          policy: Sequences of distribution params of the current policy.
          action: Sequences of actions.
          advantage: Sequences of advantages.
          length: Batch of sequence lengths.

        Returns:
          Tuple of loss tensor and summary tensor.
        """
        with tf.name_scope('policy_loss'):
            # only need to compare the kl divergence between the same policy, offense or defense
            # action.shape=(batch_size,episode_len,11,2)
            batch_size = tf.shape(action)[0]
            episode_len = action.shape.as_list()[1]
            off_action_policy_format = tf.reshape(
                action[:, :, :6], shape=[batch_size, episode_len, 12])
            def_dash_policy_format = tf.reshape(
                action[:, :, 6:], shape=[batch_size, episode_len, 10])
            action_policy_format = []
            action_policy_format.append(off_action_policy_format[:, :, 0])
            action_policy_format.append(off_action_policy_format[:, :, 1:])
            action_policy_format.append(def_dash_policy_format)

            def get_policy_gradient(category):
                return tf.exp(policy[category].log_prob(action_policy_format[category]) - old_policy[category].log_prob(action_policy_format[category]))
            off_policy_gradient = (get_policy_gradient(
                ACT['DECISION']) + get_policy_gradient(ACT['OFF_DASH'])) / 2.0  # NOTE: weighted sum maybe
            def_policy_gradient = get_policy_gradient(ACT['DEF_DASH'])
            policy_gradient = tf.where(
                self._is_optimizing_offense, off_policy_gradient, def_policy_gradient)

            def get_kl_divergence(category):
                return tf.contrib.distributions.kl_divergence(old_policy[category], policy[category])
            off_kl = (get_kl_divergence(
                ACT['DECISION']) + get_kl_divergence(ACT['OFF_DASH'])) / 2.0  # NOTE: weighted sum maybe
            def_kl = get_kl_divergence(ACT['DEF_DASH'])
            kl = tf.where(self._is_optimizing_offense, off_kl, def_kl)
            # Infinite values in the KL, even for padding frames that we mask out,
            # cause NaN gradients since TensorFlow computes gradients with respect to
            # the whole input tensor.
            kl = tf.check_numerics(kl, 'kl')
            kl = tf.reduce_mean(self._mask(kl, length), 1)

            surrogate_loss = -tf.reduce_mean(self._mask(
                policy_gradient * tf.stop_gradient(advantage), length), 1)
            surrogate_loss = tf.check_numerics(
                surrogate_loss, 'surrogate_loss')
            kl_penalty = self._penalty * kl
            cutoff_threshold = self._config.kl_target * self._config.kl_cutoff_factor
            cutoff_count = tf.reduce_sum(
                tf.cast(kl > cutoff_threshold, tf.int32))
            with tf.control_dependencies([tf.cond(
                    cutoff_count > 0,
                    lambda: tf.Print(0, [cutoff_count], 'kl cutoff! '), int)]):
                kl_cutoff = (
                    self._config.kl_cutoff_coef *
                    tf.cast(kl > cutoff_threshold, tf.float32) *
                    (kl - cutoff_threshold) ** 2)
            policy_loss = surrogate_loss + kl_penalty + kl_cutoff

            def get_entropy(category):
                return policy[category].entropy()
            # NOTE: about entropy in ACT['DECISION'], NO_OP might more likely happen then SHOOT and PASS
            off_entropy = (get_entropy(ACT['DECISION']) +
                           get_entropy(ACT['OFF_DASH'])) / 2.0  # NOTE: weighted sum maybe
            def_entropy = get_entropy(ACT['DEF_DASH'])
            entropy = tf.where(self._is_optimizing_offense,
                               off_entropy, def_entropy)
            if self._config.entropy_regularization:
                policy_loss -= self._config.entropy_regularization * entropy
            # TODO summary
            summary = tf.summary.merge([
                tf.summary.histogram('entropy', entropy),
                tf.summary.histogram('kl', kl),
                tf.summary.histogram('surrogate_loss', surrogate_loss),
                tf.summary.histogram('kl_penalty', kl_penalty),
                tf.summary.histogram('kl_cutoff', kl_cutoff),
                tf.summary.histogram('kl_penalty_combined',
                                     kl_penalty + kl_cutoff),
                tf.summary.histogram('policy_loss', policy_loss),
                tf.summary.scalar(
                    'avg_surr_loss', tf.reduce_mean(surrogate_loss)),
                tf.summary.scalar('avg_kl_penalty',
                                  tf.reduce_mean(kl_penalty)),
                tf.summary.scalar('avg_policy_loss', tf.reduce_mean(policy_loss))])
            policy_loss = tf.reduce_mean(policy_loss, 0)
            return tf.check_numerics(policy_loss, 'policy_loss'), summary

    def _adjust_penalty(self, observ, old_policy_params, length):
        """Adjust the KL policy between the behavioral and current policy.

        Compute how much the policy actually changed during the multiple
        update steps. Adjust the penalty strength for the next training phase if we
        overshot or undershot the target divergence too much.

        Args:
          observ: Sequences of observations.
          old_policy_params: Parameters of the behavioral policy.
          length: Batch of sequence lengths.

        Returns:
          Summary tensor.
        """
        old_policy = []
        for _, v in ACT.items():
            old_policy.append(self._policy_type[v](**(old_policy_params[v])))
        with tf.name_scope('adjust_penalty'):
            network = self._network(observ, length)

            def get_kl_divergence(category):
                return tf.contrib.distributions.kl_divergence(old_policy[category], network.policy[category])
            off_kl = (get_kl_divergence(
                ACT['DECISION']) + get_kl_divergence(ACT['OFF_DASH'])) / 2.0  # NOTE: weighted sum maybe
            def_kl = get_kl_divergence(ACT['DEF_DASH'])
            kl = tf.where(self._is_optimizing_offense, off_kl, def_kl)
            print_penalty = tf.Print(0, [self._penalty], 'current penalty: ')
            with tf.control_dependencies([print_penalty]):
                kl_change = tf.reduce_mean(self._mask(
                    kl,
                    length))
                kl_change = tf.Print(kl_change, [kl_change], 'kl change: ')
                maybe_increase = tf.cond(
                    kl_change > 1.3 * self._config.kl_target,
                    # pylint: disable=g-long-lambda
                    lambda: tf.Print(self._penalty.assign(
                        self._penalty * 1.5), [0], 'increase penalty '),
                    float)
                maybe_decrease = tf.cond(
                    kl_change < 0.7 * self._config.kl_target,
                    # pylint: disable=g-long-lambda
                    lambda: tf.Print(self._penalty.assign(
                        self._penalty / 1.5), [0], 'decrease penalty '),
                    float)
            with tf.control_dependencies([maybe_increase, maybe_decrease]):
                return tf.summary.merge([
                    tf.summary.scalar('kl_change', kl_change),
                    tf.summary.scalar('penalty', self._penalty)])

    def _mask(self, tensor, length, padding_value=0):
        """Set padding elements of a batch of sequences to a constant.

        Useful for setting padding elements to zero before summing along the time
        dimension, or for preventing infinite results in padding elements.

        Args:
          tensor: Tensor of sequences.
          length: Batch of sequence lengths.
          padding_value: Value to write into padding elemnts.

        Returns:
          Masked sequences.
        """
        with tf.name_scope('mask'):
            range_ = tf.range(tensor.shape[1].value)
            mask = range_[None, :] < length[:, None]
            if tensor.shape.ndims > 2:
                for _ in range(tensor.shape.ndims - 2):
                    mask = mask[..., None]
                mask = tf.tile(mask, [1, 1] + tensor.shape[2:].as_list())
            masked = tf.where(mask, tensor, padding_value *
                              tf.ones_like(tensor))
            return tf.check_numerics(masked, 'masked')


TEAM = {
    'OFFENSE': 0,
    'DEFENSE': 1
}

ACT = {
    'DECISION': 0,
    'OFF_DASH': 1,
    'DEF_DASH': 2
}
