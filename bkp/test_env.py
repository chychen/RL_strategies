from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from bball_strategies import gym_bball
import numpy as np


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


def ball_looping():
    for i in range(5):
        action = tuple((
            np.array(1),
            np.array([0.1, 0.1]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()

        no_op()
        no_op()
        no_op()
        no_op()
        no_op()

        action = tuple((
            np.array(1),
            np.array([0.0, -0.1]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()

        no_op()
        no_op()
        no_op()
        no_op()
        no_op()

        action = tuple((
            np.array(1),
            np.array([-0.1, 0.1]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()

        no_op()
        no_op()
        no_op()
        no_op()
        no_op()


def moving_around():
    for _ in range(3):
        action = tuple((
            np.array(2),
            np.array([0, 0]),
            np.array([[-2, 0], [-2, 0], [-2, 0], [-2, 0], [-2, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()
        no_op()
        no_op()
        no_op()

    no_op()
    no_op()

    for i in range(3):
        action = tuple((
            np.array(2),
            np.array([0, 0]),
            np.array([[2, 0], [2, 0], [2, 0], [2, 0], [2, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()
        no_op()
        no_op()
        no_op()


def maxspeed():
    for _ in range(10):
        action = tuple((
            np.array(2),
            np.array([0, 0]),
            np.array([[0, 0], [0, 0], [-5, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()
        no_op()


def ball_stealing():
    global env
    env = gym.make('bball-v0')
    init_positions = [
        np.array([45, 40]),
        np.array([
            [45, 10],
            [45, 40],
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype=np.float),
        np.array([
            [45, 37],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype=np.float)
    ]
    init_ball_handler_idx = 1
    env = MonitorWrapper(env,
                         init_mode=2,
                         init_positions=init_positions,
                         init_ball_handler_idx=init_ball_handler_idx,
                         if_vis_trajectory=True,
                         if_vis_visual_aid=True)
    env.reset()
    env.render()
    action = tuple((
        np.array(1),
        np.array([0, -0.1]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    ))
    _, _, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()
    for _ in range(10):
        no_op()
    env.close()


def move_offense_right():
    for _ in range(25):
        action = tuple((
            np.array(2),
            np.array([0, 0]),
            np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()
        action = tuple((
            np.array(2),
            np.array([0, 0]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()


def random_dancing():
    for _ in range(48):
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            env.render()
        print('###################### REWARD ######################', reward)


class MonitorWrapper(gym.wrappers.Monitor):
    # init_mode 0 : init by default
    def __init__(self, env, init_mode=None, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, init_ball_handler_idx=None):
        super(MonitorWrapper, self).__init__(env=env, directory='./test/',
                                             video_callable=lambda count: count % 1 == 0, force=True)
        env.init_mode = init_mode
        env.if_vis_trajectory = if_vis_trajectory
        env.if_vis_visual_aid = if_vis_visual_aid
        env.init_positions = init_positions
        env.init_ball_handler_idx = init_ball_handler_idx


def rewards():
    global env
    env = gym.make('bball-v0')
    init_positions = [
        np.array([70, 40]),
        np.array([
            [70, 40],
            [71, 38],
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype=np.float),
        np.array([
            [76, 32],
            [70, 30],
            [60, 40],
            [0, 0],
            [0, 0]
        ], dtype=np.float)
    ]
    init_ball_handler_idx = 0
    env = MonitorWrapper(env,
                         init_mode=2,
                         init_positions=init_positions,
                         init_ball_handler_idx=init_ball_handler_idx,
                         if_vis_trajectory=False,
                         if_vis_visual_aid=True)
    env.reset()
    env.render()
    action = tuple((  # shoot decision
        np.array(0),
        np.array(0),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    ))
    _, reward, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()
        env.render()
    input()
    env.close()


def main():
    global env
    env = gym.make('bball-v0')
    env = MonitorWrapper(env,
                         init_mode=0,
                         if_vis_trajectory=False,
                         if_vis_visual_aid=True)
    obs = env.reset()
    print(obs)
    print('#########################################')
    env.render()

    # DEMO script
    # ball_looping()
    # moving_around()
    # maxspeed()
    move_offense_right()
    # random_dancing()

    env.close()


if __name__ == '__main__':
    main()
    # rewards()
    # ball_stealing()