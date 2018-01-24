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
    def __init__(self, env, if_init_by_default=False, if_vis_trajectory=False, if_vis_visual_aid=False):
        super(MonitorWrapper, self).__init__(env=env, directory='./test/',
                                             video_callable=lambda count: count % 1 == 0, force=True)
        self.if_init_by_default = if_init_by_default
        self.if_vis_trajectory = if_vis_trajectory
        self.if_vis_visual_aid = if_vis_visual_aid

    def reset(self):
        return self._reset(if_init_by_default=self.if_init_by_default)

    def render(self):
        return self.env._render(if_vis_trajectory=self.if_vis_trajectory, if_vis_visual_aid=self.if_vis_visual_aid)


def main():
    global env
    env = gym.make('bball-v0')
    env = MonitorWrapper(env, if_init_by_default=True,
                         if_vis_trajectory=True,
                         if_vis_visual_aid=True)
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
