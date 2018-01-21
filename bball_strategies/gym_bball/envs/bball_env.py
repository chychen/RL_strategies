""" Define the basketball game environment with continuous state spaces and continuous, discrete action spaces, containing two agents:
1. offensive team (and ball):
    - actions:
        * speeds (continuous): for each player and ball
            sampled from policy output's probability distribution
        * decision (discrete): one hot vector, shape=[3]
            choosed by policy softmax layers' max output
            ` shoot: after moving, calculate the open score, episode ends.
            ` pass: move players and pass the ball at same time.
            ` no-op: do nothing but moving, excluding the ball.
        * (maybe the communication protocol to encourage set offense)
    - rewards (open score):
        * the farther to the cloest defender the better
        * the closer to the basket the better
        * (maybe encourage passing decision)
2. defensive team:
    - actions:
        * speeds (continuous): for each player
            sampled from policy output's probability distribution
        * (maybe the communication protocol to encourage set defense)
    - rewards (neg open score):
        * the closer to shooting man the better
        * the more defenders closest to ball-holder the better
        * possess the ball in the end of the episode
        * (maybe encourage person-to-person defenses)

Environment:
    - The task is episodic, and an episode ends when one of four events occurs:
        * A shoot decision is made,
        * The ball is out of bounds,
        * A defender gets possession of the ball,
        * A max time limit for the episode is reached.
    - Constraints:
        * Catching/Stealing Ball
        * Collision
    - Rewards:
        * Offensive team:
        * Defensive team:
court size: 94 by 50 (feet)
1 feet = 30.48 cm = 0.3048 m
1 m = 3.2808 feet
# analized from real NBA dataset
(maximum = mean + 3 * stddev)
FPS = 5 
### ball TODO
maximum speed =  m/s =  feet/s =  feet/frame
maximum accerlation =  m/s^2 =  feet/s^2 =  feet/frame
### offensive players TODO
maximum speed =  m/s =  feet/s =  feet/frame
maximum accerlation =  m/s^2 =  feet/s^2 =  feet/frame
### defensive players TODO
maximum speed =  m/s =  feet/s =  feet/frame
maximum accerlation =  m/s^2 =  feet/s^2 =  feet/frame
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
from os import path
"""
# TODO
# check what is the screen range?
# screen_range = circle with radius 2.0 feets (about 0.6 meter)
"""


class BBallEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # for render()
        self.viewer = None
        self.def_pl_transforms = []
        self.off_pl_transforms = []
        self.ball_transform = None
        self.fps = 5.0
        # court information
        self.court_length = 94
        self.court_width = 50
        self.left_basket_pos = [0 + 5.25, 25]
        self.right_basket_pos = [94 - 5.25, 25]
        # physics limitations TODO per frame
        self.pl_max_speed = 1
        self.pl_max_acc = 1
        self.ba_max_speed = 1
        self.ba_max_acc = 1
        self.screen_radius = 2.0
        self.if_ball_flying = False
        # rl
        self.state = None

        # must define properties
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

        # random seed
        self._seed()

    def _step(self, action):
        # step by action
        # if collision
        # if ball flying
        # if ball caught/stolen
        # if episode ended
        # A shoot decision is made,
        # The ball is out of bounds.
        # A defender gets possession of the ball.
        # A max time limit for the episode is reached.
        # if action exceed the physics limits
        ...

    def _reset(self):
        """ random init positions in the right half court
        1. init offensive team randomlu
        2. add defensive team next to each offensive player in the basket side.
        """
        off_players_pos = self.np_random_generator.uniform(
            low=[self.court_length // 2, 0], high=[self.court_length, self.court_width], size=[5, 2])

        def_players_pos = np.array(off_players_pos, copy=True)
        vec = self.right_basket_pos - off_players_pos
        vec_length = np.sqrt(np.sum(vec * vec, axis=1))
        u_vec = vec / np.stack([vec_length, vec_length], axis=1)
        def_players_pos = def_players_pos + u_vec * self.screen_radius

        ball_idx = np.floor(self.np_random_generator.uniform(
            low=0.0, high=5.0)).astype(np.int)
        ball_pos = np.array(off_players_pos[ball_idx, :], copy=True)

        self.state = np.array([ball_pos, off_players_pos, def_players_pos])
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            viewer = rendering.Viewer(940, 500)
            viewer.set_bounds(0, 94, 0, 50)  # feet
            # background img
            fname = path.join(path.dirname(__file__), "fullcourt.png")
            img = rendering.Image(fname, 94, 50)
            imgtrans = rendering.Transform(translation=(47.0, 25.0))
            img.add_attr(imgtrans)
            viewer.add_geom(img)
            # defensive players
            for _ in range(5):
                def_player = rendering.make_circle(radius=2.)
                def_player.set_color(0, 0, 1)
                def_trans = rendering.Transform()
                self.def_pl_transforms.append(def_trans)
                def_player.add_attr(def_trans)
                viewer.add_geom(def_player)
            # offensive players
            for _ in range(5):
                off_player = rendering.make_circle(radius=2.)
                off_player.set_color(1, 0, 0)
                off_trans = rendering.Transform()
                self.off_pl_transforms.append(off_trans)
                off_player.add_attr(off_trans)
                viewer.add_geom(off_player)
            # ball
            ball = rendering.make_circle(radius=1.)
            ball.set_color(0, 1, 0)
            ball_trans = rendering.Transform()
            self.ball_transform = ball_trans
            ball.add_attr(ball_trans)
            viewer.add_geom(ball)

        ### set translations ###
        # defensive players
        for trans, pos in zip(self.def_pl_transforms, self.state[STATE_LOOKUP['DEFENSE']]):
            trans.set_translation(pos[0], pos[1])
        # offensive players
        for trans, pos in zip(self.off_pl_transforms, self.state[STATE_LOOKUP['OFFENSE']]):
            trans.set_translation(pos[0], pos[1])
        # ball
        ball_pos = self.state[STATE_LOOKUP['BALL']]
        self.ball_transform.set_translation(ball_pos[0], ball_pos[1])

        return viewer.render(return_rgb_array=mode == 'rgb_array')

    # def _close(self):
    #     pass

    def _seed(self, seed=None):
        # seeding use very strong random seed generated by os
        self.np_random_generator, seed = seeding.np_random(seed)
        return [seed]

    def _set_action_space(self):
        """
        Return
        ------
        Tuple(Discrete(2), Discrete(3), Box(2,), Box(5, 2), Box(5, 2))
        """
        return spaces.Tuple((
            spaces.Discrete(2),  # offense or defense
            spaces.Discrete(3),
            # ball acc
            spaces.Box(
                low=np.zeros([2, ]) - self.ba_max_speed, high=np.zeros([2, ]) + self.ba_max_speed),
            # offense player acc
            spaces.Box(
                low=np.zeros([5, 2]) - self.pl_max_speed, high=np.zeros([5, 2]) + self.pl_max_speed),
            # defense player acc
            spaces.Box(
                low=np.zeros([5, 2]) - self.pl_max_speed, high=np.zeros([5, 2]) + self.pl_max_speed)
        ))

    def _set_observation_space(self):
        """ positions only valid in right-half court
        Return
        ------
        Tuple(Box(2,), Box(5, 2), Box(5, 2))
        """
        return spaces.Tuple((
            # ball position
            spaces.Box(low=np.array([self.court_length // 2, 0]),
                       high=np.array([self.court_length, self.court_width])),
            # offense player positions
            spaces.Box(low=np.array([[self.court_length // 2, 0] for _ in range(5)]),
                       high=np.array([[self.court_length, self.court_width] for _ in range(5)])),
            # defense player positions
            spaces.Box(low=np.array([[self.court_length // 2, 0] for _ in range(5)]),
                       high=np.array([[self.court_length, self.court_width] for _ in range(5)]))
        ))

STATE_LOOKUP = {
    # Tuple(Box(2,), Box(5, 2), Box(5, 2))
    'BALL': 0,
    'OFFENSE': 1,
    'DEFENSE': 2
}

ACTION_LOOKUP = {
    # Tuple(Discrete(2), Discrete(3), Box(2,), Box(5, 2), Box(5, 2))
    'FLAG': 0,
    'ACTION': 1,
    'BALL': 2,
    'OFFENSE': 3,
    'DEFENSE': 4
}
