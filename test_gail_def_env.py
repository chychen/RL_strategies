from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from bball_strategies import gym_bball
import numpy as np
import h5py


def no_op():
    action = tuple((
        np.array(2),
        np.array([0, 0]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    ))
    _, _, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()
        env.render()


def random_dancing():
    for _ in range(100000000):
        action = env.action_space.sample()
        _, reward, done, infos = env.step(action)
        env.render()
        
        if done:
            env.reset()
            env.render()
        print('###################### REWARD ######################', reward)


class MonitorWrapper(gym.wrappers.Monitor):
    # init_mode 0 : init by default
    def __init__(self, env, data=h5py.File('bball_strategies/data/OrderedGAILTransitionData_52.hdf5', 'r'), init_mode=None, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, init_ball_handler_idx=None):
        super(MonitorWrapper, self).__init__(env=env, directory='./test/',
                                             video_callable=lambda count: True, force=True)
        env.init_mode = init_mode
        env.data = data
        env.time_limit = 50
        env.if_vis_trajectory = if_vis_trajectory
        env.if_vis_visual_aid = if_vis_visual_aid
        env.init_positions = init_positions
        env.init_ball_handler_idx = init_ball_handler_idx

    def __getattr__(self, name):
        return getattr(self._env, name)


def main():
    global env
    env = gym.make('bball_gail_def-v0')
    env = MonitorWrapper(env,
                         init_mode=1,
                         if_vis_trajectory=False,
                         if_vis_visual_aid=True)
    obs = env.reset()
    print(obs)
    print('#########################################')
    env.render()

    random_dancing()

    env.close()


if __name__ == '__main__':
    main()