import gym
from bball_strategies import gym_bball
import numpy as np


def no_op():
    action = tuple((
        # np.array([1, 0]),
        np.array(2),
        np.array(0),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    ))
    _, _, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()


def ball_looping():
    for i in range(5):
        # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        action = tuple((
            np.array(1),
            np.array(np.pi / 4),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()

        no_op()
        no_op()

        action = tuple((
            np.array(1),
            np.array(-np.pi / 2),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()

        no_op()
        no_op()
        no_op()

        action = tuple((
            np.array(1),
            np.array(np.pi * 3 / 4),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()

        no_op()
        no_op()


def moving_around():
    for i in range(3):
        # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        action = tuple((
            np.array(2),
            np.array(0),
            np.array(
                [[2, np.pi], [2, np.pi], [2, np.pi], [2, np.pi], [2, np.pi]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()

    no_op()
    no_op()
    no_op()

    no_op()
    no_op()
    no_op()

    for i in range(3):
        # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        action = tuple((
            np.array(2),
            np.array(0),
            np.array(
                [[2, 0], [2, 0], [2, 0], [2, 0], [2, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()


def maxspeed():
    for _ in range(10):
        action = tuple((
            np.array(2),
            np.array(0),
            np.array([[0, 0], [0, 0], [5, np.pi], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
    no_op()


def ball_stealing():
    # off_players_pos = np.array([
    #     [45, 10],
    #     [45, 40],
    #     [0, 0],
    #     [0, 0],
    #     [0, 0]
    # ], dtype=np.float)
    # ball_handler_idx = 1
    # def_players_pos[0] = [45, 37]

    action = tuple((
        np.array(1),
        np.array(-np.pi / 2),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    ))
    _, _, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()
    for _ in range(10):
        no_op()


def move_offense_right():
    for _ in range(15):
        action = tuple((
            np.array(2),
            np.array(0),
            np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()


def random_dancing():
    for _ in range(20):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()


class MonitorWrapper(gym.wrappers.Monitor):
    def __init__(self, env, if_init_by_default=False, if_vis_trajectory=False, if_vis_visual_aid=False):
        super(MonitorWrapper, self).__init__(env=env, directory='./test/',
                                             video_callable=lambda count: count % 1 == 0, force=True)
        env.if_init_by_default = if_init_by_default
        env.if_vis_trajectory = if_vis_trajectory
        env.if_vis_visual_aid = if_vis_visual_aid


def main():
    global env
    env = gym.make('bball-v0')
    env = MonitorWrapper(env, if_init_by_default=True,
                         if_vis_trajectory=False,
                         if_vis_visual_aid=True)
    env.reset()
    env.render()

    # DEMO script
    # ball_looping()
    # moving_around()
    # maxspeed()
    # ball_stealing()
    # move_offense_right()
    random_dancing()
    # terminal_conditions()
    # rewards()

    env.close()


if __name__ == '__main__':
    main()

# gather：根據一個list來取用目標
# python 使用+=很可能會有不明錯誤！！！儘量避免
# numpy 有outer機制，計算cross-op可避免使用迴圈
# np.op.outer(), i.e. np.subtract.outer()
# np.inner, can broadcast itsef
