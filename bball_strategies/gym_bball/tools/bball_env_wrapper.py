"""Wrappers for BBall environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from gym import spaces
from bball_strategies.gym_bball import tools


class BBallWrapper(object):

    def __init__(self, env, data, if_clip=True, if_norm_obs=True, if_norm_act=True, init_mode=1, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, fps=5, time_limit=240, if_back_real=False):
        """
        Args
        ----
        init_mode : 0->default, 1->dataset, 2->customized

        1. Convert to 32 bit (just make sure)
        2. Normalize obs and denormalize action
        3. Clip actions by env settings

        time_limit : default value = 24 sec * 5 fps * 2 teams
        """
        self._env = env
        self._env.init_mode = init_mode
        self._env.if_vis_trajectory = if_vis_trajectory
        self._env.if_vis_visual_aid = if_vis_visual_aid
        self._env.init_positions = init_positions
        self._env.FPS = fps
        self._env.data = data
        self._env.time_limit = time_limit
        self._env.if_back_real = if_back_real

        if if_clip:
            self._env = ClipAction(self._env)
        self._env = RangeNormalize(
            self._env, observ=if_norm_obs, action=if_norm_act)
        self._env = ConvertTo32Bit(self._env)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        return self._env.reset()

    @property
    def data(self):
        return self._env.data

    @data.setter
    def data(self, value):
        self._env.data = value


class ClipAction(object):
    """Clip out of range actions to the action space of the environment."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        """ clip action to acceptable range before step()
        action : shape=[Discrete(3), Box(), Box(5, 2), Box(5, 2)]
        """
        action_space = self._env.action_space  # Tuple(Discrete(3), Box(), Box(5, 2), Box(5, 2))
        for i, space in enumerate(action_space):
            if isinstance(space, spaces.Discrete):  # no need to clip
                pass
            elif isinstance(space, spaces.Box):
                action[i] = np.clip(action[i], space.low, space.high)
            else:
                raise ValueError(
                    'action space is not defined, {}'.format(action[i]))
        return self._env.step(action)

    @property
    def data(self):
        return self._env.data

    @data.setter
    def data(self, value):
        self._env.data = value


class RangeNormalize(object):
    """ Normalize the specialized observation and action ranges to [-1, 1]."""

    def __init__(self, env, observ=None, action=None):
        self._env = env
        # validate
        self._should_normalize_observ = (
            observ is not False and self._is_finite(self._env.observation_space))
        if observ is True and not self._should_normalize_observ:
            raise ValueError('Cannot normalize infinite observation range.')
        if observ is None and not self._should_normalize_observ:
            tf.logging.info('Not normalizing infinite observation range.')
        self._should_normalize_action = (
            action is not False and self._is_finite(self._env.action_space))
        if action is True and not self._should_normalize_action:
            raise ValueError('Cannot normalize infinite action range.')
        if action is None and not self._should_normalize_action:
            tf.logging.info('Not normalizing infinite action range.')

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        space = self._env.observation_space
        if not self._should_normalize_observ:
            return space
        return spaces.Box(-np.ones(space.shape), np.ones(space.shape), dtype=np.float32)

    @property
    def action_space(self):
        space = self._env.action_space
        if not self._should_normalize_action:
            return space
        if isinstance(space, tools.ActTuple):
            return tools.ActTuple((
                spaces.Discrete(3),  # offensive decision
                # ball theta
                spaces.Box(-np.ones(space[1].shape),
                        np.ones(space[1].shape), dtype=np.float32),
                # offense player DASH(power, direction)
                spaces.Box(-np.ones(space[2].shape),
                        np.ones(space[2].shape), dtype=np.float32),
                # defense player DASH(power, direction)
                spaces.Box(-np.ones(space[3].shape),
                        np.ones(space[3].shape), dtype=np.float32)
            ))
        elif isinstance(space, tools.NDefActTuple):
            return tools.NDefActTuple((
                spaces.Discrete(3),  # offensive decision
                # ball theta
                spaces.Box(-np.ones(space[1].shape),
                        np.ones(space[1].shape), dtype=np.float32),
                # offense player DASH(power, direction)
                spaces.Box(-np.ones(space[2].shape),
                        np.ones(space[2].shape), dtype=np.float32)
            ))

    def step(self, action):
        if self._should_normalize_action:
            action = self._denormalize_action(action)
        observ, reward, done, info = self._env.step(action)
        if self._should_normalize_observ:
            observ = self._normalize_observ(observ)
        return observ, reward, done, info

    def reset(self):
        observ = self._env.reset()
        if self._should_normalize_observ:
            observ = self._normalize_observ(observ)
        return observ

    def _denormalize_action(self, action):
        # skip discrete item (self._env.action_space[0])
        for i in range(1, len(action)):
            min_ = self._env.action_space[i].low
            max_ = self._env.action_space[i].high
            action[i] = (action[i] + 1) / 2 * (max_ - min_) + min_
        return action

    def _normalize_observ(self, observ):
        min_ = self._env.observation_space.low
        max_ = self._env.observation_space.high
        observ = 2 * (observ - min_) / (max_ - min_) - 1
        return observ

    def _is_finite(self, space):
        check = True
        if hasattr(space, 'low'):  # observation is Box
            check = np.isfinite(space.low).all(
            ) and np.isfinite(space.high).all()
        else:  # action is ActTuple
            check = check and space[0].dtype == np.int64  # Discrete
            for i in range(1, space.len):
                check = check and np.isfinite(space[i].low).all(
                ) and np.isfinite(space[i].high).all()  # Box
        return check

    @property
    def data(self):
        return self._env.data

    @data.setter
    def data(self, value):
        self._env.data = value


class ConvertTo32Bit(object):
    """Convert data types of an OpenAI Gym environment to 32 bit."""

    def __init__(self, env):
        """Convert data types of an OpenAI Gym environment to 32 bit.

        Args:
          env: OpenAI Gym environment.
        """
        self._env = env

    def __getattr__(self, name):
        """Forward unimplemented attributes to the original environment.

        Args:
          name: Attribute that was accessed.

        Returns:
          Value behind the attribute name in the wrapped environment.
        """
        return getattr(self._env, name)

    def step(self, action):
        """Forward action to the wrapped environment.

        Args:
          action: Action to apply to the environment.

        Raises:
          ValueError: Invalid action.

        Returns:
          Converted observation, converted reward, done flag, and info object.
        """
        observ, reward, done, info = self._env.step(action)
        observ = self._convert_observ(observ)
        reward = self._convert_reward(reward)
        return observ, reward, done, info

    def reset(self):
        """Reset the environment and convert the resulting observation.

        Returns:
          Converted observation.
        """
        observ = self._env.reset()
        observ = self._convert_observ(observ)
        return observ

    def _convert_observ(self, observ):
        """Convert the observation to 32 bits.

        Args:
          observ: Numpy observation.

        Raises:
          ValueError: Observation contains infinite values.

        Returns:
          Numpy observation with 32-bit data type.
        """
        if not np.isfinite(observ).all():
            raise ValueError('Infinite observation encountered.')
        if observ.dtype == np.float64:
            return observ.astype(np.float32)
        if observ.dtype == np.int64:
            return observ.astype(np.int32)
        return observ

    def _convert_reward(self, reward):
        """Convert the reward to 32 bits.

        Args:
          reward: Numpy reward.

        Raises:
          ValueError: Rewards contain infinite values.

        Returns:
          Numpy reward with 32-bit data type.
        """
        if not np.isfinite(reward).all():
            raise ValueError('Infinite reward encountered.')
        return np.array(reward, dtype=np.float32)

    @property
    def data(self):
        return self._env.data

    @data.setter
    def data(self, value):
        self._env.data = value
