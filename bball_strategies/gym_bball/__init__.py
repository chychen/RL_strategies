""" Environment settings for bball strategies"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register

register(
    id='bball-v0',
    entry_point='bball_strategies.gym_bball.envs:BBallEnv',
)

register(
    id='bball-pretrain-v0',
    entry_point='bball_strategies.gym_bball.envs:BBallPretrainEnv',
)

register(
    id='bball-ndef-v0',
    entry_point='bball_strategies.gym_bball.envs:BBallNDefEnv',
)

register(
    id='bball-ndef-bspeed-v0',
    entry_point='bball_strategies.gym_bball.envs:BBallNDefBSpeedEnv',
)

register(
    id='bball_gail_def-v0',
    entry_point='bball_strategies.gym_bball.envs:BBallGailDefEnv',
)
