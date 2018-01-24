import gym
from bball_strategies import gym_bball
import numpy as np


def no_op():
    action = tuple((
        np.array([1, 0]),
        np.array([0, 0, 1]),
        np.array(0),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    ))
    env.step(action)
    env.render()


def ball_passing():
    for i in range(5):
        # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        action = tuple((
            np.array([1, 0]),
            np.array([0, 1, 0]),
            np.array(np.pi / 4),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()

        no_op()
        no_op()

        action = tuple((
            np.array([1, 0]),
            np.array([0, 1, 0]),
            np.array(-np.pi / 2),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()

        no_op()
        no_op()
        no_op()

        action = tuple((
            np.array([1, 0]),
            np.array([0, 1, 0]),
            np.array(np.pi * 3 / 4),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()

        no_op()
        no_op()


def moving_around():
    for i in range(3):
        # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        action = tuple((
            np.array([1, 0]),
            np.array([0, 0, 1]),
            np.array(0),
            np.array(
                [[1, np.pi / 2], [1, np.pi / 2], [1, np.pi / 2], [1, np.pi / 2], [1, np.pi / 2]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()

    for i in range(6):
        # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        action = tuple((
            np.array([1, 0]),
            np.array([0, 0, 1]),
            np.array(0),
            np.array(
                [[1, -np.pi / 2], [1, -np.pi / 2], [1, -np.pi / 2], [1, -np.pi / 2], [1, -np.pi / 2]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()

    no_op()
    no_op()
    no_op()

    for i in range(3):
        # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        action = tuple((
            np.array([1, 0]),
            np.array([0, 0, 1]),
            np.array(0),
            np.array(
                [[1, np.pi / 2], [1, np.pi / 2], [1, np.pi / 2], [1, np.pi / 2], [1, np.pi / 2]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()

def collision():
    pass


def main():
    global env
    env = gym.make('bball-v0')
    env = gym.wrappers.Monitor(
        env, './test/', video_callable=lambda count: count % 1 == 0, force=True)
    env._reset(is_default_init=True) 
    env.render()
    # DEMO script
    # ball_passing()
    # moving_around()
    ball_stealing()
    # maxspeed()
    # collision()
    # random_dancing()
    # rewards()

    env.close()


if __name__ == '__main__':
    main()

# gather：根據一個list來取用目標
# python 使用+=很可能會有不明錯誤！！！儘量避免
