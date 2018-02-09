"""Wrappers for BBall environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces


class BBallWrapper(object):

    def __init__(self, env, init_mode=0, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, init_ball_handler_idx=None, fps=5, time_limit=240):
        """
        1. Convert to 32 bit
        2. Clip actions
        3. Normalize obs and action
        time_limit : default value = 24 sec * 5 fps * 2 teams
        """
        self._env = env
        self._env.init_mode = init_mode
        self._env.if_vis_trajectory = if_vis_trajectory
        self._env.if_vis_visual_aid = if_vis_visual_aid
        self._env.init_positions = init_positions
        self._env.init_ball_handler_idx = init_ball_handler_idx
        self._env.FPS = fps
        self._env.time_limit = time_limit

        self._env = ConvertTo32Bit(self._env)
        # TODO normalization

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        return self._env.reset()

    @property
    def action_space(self):
        """
        to walk around for value range validation, we will force clip action value before step()
        """
        shape = self._env.action_space.shape  # (11,2)
        return spaces.Box(-np.inf * np.ones(shape), np.inf * np.ones(shape), dtype=np.float32)

    def step(self, action):
        """
        action : shape=[Discrete(3), Box(), Box(5, 2), Box(5, 2)]
        """
        action_space = self._env.action_space  # Tuple(Discrete(3), Box(), Box(5, 2), Box(5, 2))
        for i, space in enumerate(action_space):
            if isinstance(space, spaces.Discrete):
                pass
            elif isinstance(space, spaces.Box):
                action[i] = np.clip(action[i], space.low, space.high)
            else:
                raise ValueError(
                    'action space is not defined, {}'.format(action[i]))
        return self._env.step(action)


# class RangeNormalize(object):
#     """ TODO Normalize the specialized observation and action ranges to [-1, 1]."""

#     def __init__(self, env, observ=None, action=None):
#         self._env = env
#         self._should_normalize_observ = (
#             observ is not False and self._is_finite(self._env.observation_space))
#         if observ is True and not self._should_normalize_observ:
#             raise ValueError('Cannot normalize infinite observation range.')
#         if observ is None and not self._should_normalize_observ:
#             tf.logging.info('Not normalizing infinite observation range.')
#         self._should_normalize_action = (
#             action is not False and self._is_finite(self._env.action_space))
#         if action is True and not self._should_normalize_action:
#             raise ValueError('Cannot normalize infinite action range.')
#         if action is None and not self._should_normalize_action:
#             tf.logging.info('Not normalizing infinite action range.')

#     def __getattr__(self, name):
#         return getattr(self._env, name)

#     @property
#     def observation_space(self):
#         space = self._env.observation_space
#         if not self._should_normalize_observ:
#             return space
#         return gym.spaces.Box(-np.ones(space.shape), np.ones(space.shape))

#     @property
#     def action_space(self):
#         space = self._env.action_space
#         if not self._should_normalize_action:
#             return space
#         return gym.spaces.Box(-np.ones(space.shape), np.ones(space.shape))

#     def step(self, action):
#         if self._should_normalize_action:
#             action = self._denormalize_action(action)
#         observ, reward, done, info = self._env.step(action)
#         if self._should_normalize_observ:
#             observ = self._normalize_observ(observ)
#         return observ, reward, done, info

#     def reset(self):
#         observ = self._env.reset()
#         if self._should_normalize_observ:
#             observ = self._normalize_observ(observ)
#         return observ

#     def _denormalize_action(self, action):
#         min_ = self._env.action_space.low
#         max_ = self._env.action_space.high
#         action = (action + 1) / 2 * (max_ - min_) + min_
#         return action

#     def _normalize_observ(self, observ):
#         min_ = self._env.observation_space.low
#         max_ = self._env.observation_space.high
#         observ = 2 * (observ - min_) / (max_ - min_) - 1
#         return observ

#     def _is_finite(self, space):
#         return np.isfinite(space.low).all() and np.isfinite(space.high).all()


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
