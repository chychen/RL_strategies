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
                [[2, np.pi], [2, np.pi], [2, np.pi], [2, np.pi], [2, np.pi]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()

    no_op()
    no_op()
    no_op()

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
                [[2, 0], [2, 0], [2, 0], [2, 0], [2, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()


def maxspeed():
    for _ in range(10):
        action = tuple((
            np.array([1, 0]),
            np.array([0, 0, 1]),
            np.array(0),
            np.array([[0, 0], [0, 0], [5, np.pi], [0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        ))
        env.step(action)
        env.render()
    no_op()


def collision():
    pass


class MonitorWrapper(gym.wrappers.Monitor):
    def __init__(self, env, is_default_init=False, enable_trajectory=False):
        super(MonitorWrapper, self).__init__(env=env, directory='./test/',
                                             video_callable=lambda count: count % 1 == 0, force=True)
        self.is_default_init = is_default_init
        self.enable_trajectory = enable_trajectory

    def reset(self):
        return self._reset(is_default_init=self.is_default_init)

    def render(self):
        return self.env._render(enable_trajectory=self.enable_trajectory)


def main():
    global env
    env = gym.make('bball-v0')
    env = MonitorWrapper(env, is_default_init=True,
                         enable_trajectory=True)
    env.reset()
    env.render()

    # DEMO script
    # ball_passing()
    # moving_around()
    # maxspeed()
    ball_stealing()
    # collision()
    # random_dancing()
    # rewards()

    env.close()


if __name__ == '__main__':
    main()

# gather：根據一個list來取用目標
# python 使用+=很可能會有不明錯誤！！！儘量避免
