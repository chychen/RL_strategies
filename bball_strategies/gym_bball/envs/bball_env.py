
""" Define the basketball game environment with continuous state spaces and continuous, discrete action spaces, containing two agents:
1. offensive team (and ball):
    - actions:
        * DASH (power, degree): move the agents with power and direction,
            for each player and ball sampled from policy output's probability distribution
        * decision (discrete): one hot vector, shape=[3]
            choosed by policy softmax layers' max output
            ` SHOOT: after moving, calculate the open score, episode ends.
            ` PASS: move players after pass the ball.
            ` NO_OP: do nothing but moving, excluding the ball.
        * (maybe the communication protocol to encourage set offense)
    - rewards (open score):
        * the farther to the cloest defender the better
        * the closer to the basket the better
        * (maybe encourage passing decision)
2. defensive team:
    - actions:
        * DASH (power, degree): move the agents with power and direction,
            for each player and ball sampled from policy output's probability distribution
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
all analized results are recorded in bball_strategies/analysis/physics_limitation.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
from os import path
import copy
import sys
from bball_strategies.gym_bball import tools
"""
# NOTE
# 把一些環境資訊物件化
# check if the screen range reasonable?
# screen_radius = circle with radius 2.0 feets (about 0.61 meter)
# wingspan_radius = circle with radius 3.5 feets (about 1.06 meter)
# stolen_radius = circle with radius 5.0 feets (about 1.52 meter)

    self.screen_radius = 2.0 * 2
    self.wingspan_radius = 3.5
    self.stolen_radius = 5.0
    self.pl_max_speed = 38.9379818754 / FPS
    # cost at least one second to cross over the opponent
    self.pl_collision_speed = self.screen_radius / FPS
    self.pl_max_power = 24.5810950984 / FPS
    self.ball_passing_speed = 30.0 / FPS

note :
- velocity: scalar with direction
- speed: scalar only
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FPS = 5.0


def length(vec, axis=1):
    return np.sqrt(np.sum(vec * vec, axis=axis))


def distance(pos_a, pos_b, axis=1):
    vec = pos_b - pos_a
    return length(vec, axis=axis)


class BBallEnv(gym.Env):
    """
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS * 2.0
    }

    def __init__(self):
        # config might be setup by Wrapper
        self.init_mode = None
        self.init_positions = None  # if_init_by_input
        self.init_ball_handler_idx = None  # if_init_by_input
        self.if_vis_trajectory = False
        self.if_vis_visual_aid = False
        # for render()
        self.viewer = None
        self.def_pl_transforms = []
        self.off_pl_transforms = []
        self.ball_transform = None
        # court information
        self.court_length = 94
        self.court_width = 50
        self.three_point_distance = 24.0  # 23.75 for NBA
        self.time_limit = 24 * FPS * 2
        self.left_basket_pos = [0 + 5.25, 25]
        self.right_basket_pos = [94 - 5.25, 25]
        # this is the distance between two players' center position
        self.screen_radius = 2.0 * 2
        self.wingspan_radius = 3.5
        self.stolen_radius = 5.0
        # physics limitations per frame
        self.ball_passing_speed = 40.0 / FPS
        self.pl_max_speed = 38.9379818754 / FPS
        # cost at least one second to cross over the opponent
        self.pl_collision_speed = self.screen_radius / FPS
        self.pl_max_power = 24.5810950984 / FPS
        # reward
        self.len2pi_weight = np.pi / 8  # 1 feet <-> 45 degrees
        self.max_reward = 100.0

        # Env information
        self.states = States()
        self.buffer_size = 5  # 5 frames = 1 second = 10 steps

        # must define properties
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

        # random seed
        self.seed()

    def step(self, action):
        """
        rewards value is only designed for offense agent.
        the negative rewards of offense could be reckoned as the rewards of defense. 

        Returns
        -------
        observation (object) : agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean) : whether the episode has ended, in which case further step() calls will return undefined results
        info (dict) : contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        decision = action[ACTION_LOOKUP['DECISION']]
        ball_pass_dir = action[ACTION_LOOKUP['BALL']]
        off_pl_dash = action[ACTION_LOOKUP['OFF_DASH']]
        def_pl_dash = action[ACTION_LOOKUP['DEF_DASH']]
        if self.states.turn == FLAG_LOOKUP['OFFENSE']:
            # logger.debug('[TURN] OFFENSE')
            self._update_player_state(
                off_pl_dash, self.states.vels, STATE_LOOKUP['OFFENSE'])
            # update ball state
            self._update_ball_state(decision, ball_pass_dir)
        elif self.states.turn == FLAG_LOOKUP['DEFENSE']:
            # logger.debug('[TURN] DEFENSE')
            self._update_player_state(
                def_pl_dash, self.states.vels, STATE_LOOKUP['DEFENSE'])

        # check if meets termination condition TODO
        if decision == DESICION_LOOKUP['SHOOT'] and self.states.steps >= 5*FPS*2:
            self.states.update_status(done=True, status=STATUS_LOOKUP['SHOOT'])
        # OOB
        oob_padding = 3
        if self.states.ball_position[0] >= self.court_length + oob_padding or self.states.ball_position[0] < self.court_length / 2 - oob_padding or self.states.ball_position[1] >= self.court_width + oob_padding or self.states.ball_position[1] < 0.0 - oob_padding:
            self.states.update_status(done=True, status=STATUS_LOOKUP['OOB'])
        # OOT
        if self.states.steps >= self.time_limit:
            self.states.update_status(done=True, status=STATUS_LOOKUP['OOT'])
        # termination conditions
        reward = 0.0
        if self.states.done:
            if self.states.status == STATUS_LOOKUP['SHOOT']:
                logger.debug('[GAME OVER], A shoot decision is made')
                if self.states.ball_handler_idx is None:
                    logger.debug('[GAME OVER], No ball handler ...')
                    reward = -1.0
                else:
                    reward = self._calculate_reward()
                pass
            elif self.states.status == STATUS_LOOKUP['CAPTURED']:
                logger.debug(
                    '[GAME OVER], A defender gets possession of the ball')
                reward = -1.0
                pass
            elif self.states.status == STATUS_LOOKUP['OOB']:
                logger.debug('[GAME OVER], The ball is out of bounds.')
                reward = -1.0
                pass
            elif self.states.status == STATUS_LOOKUP['OOT']:
                logger.debug(
                    '[GAME OVER], Max time limit for the episode is reached')
                reward = -1.0
                pass
        else:
            if self.states.status == STATUS_LOOKUP['CATCH']:
                logger.debug('[GAME STATUS] Successfully Pass :D')
                reward = 1.0
                pass

        # update env information
        self.states.take_turn()

        return self._get_obs(), reward, self.states.done, dict(turn=self.states.turn)

    def _get_obs(self):
        """
        ### return observation shape
        5 : 5 frames
        14 : ball(1) + offense(5) + defense(5) + basket(1) + ball_boundry(2)
        2 : x and y positions
        """
        obs = np.empty(shape=(self.buffer_size, 14, 2), dtype=np.float32)
        for i in range(self.buffer_size):
            obs[i] = np.concatenate([
                np.expand_dims(
                    self.states.buffer_positions[i, STATE_LOOKUP['BALL']], axis=0),
                self.states.buffer_positions[i, STATE_LOOKUP['OFFENSE']],
                self.states.buffer_positions[i, STATE_LOOKUP['DEFENSE']],
                np.expand_dims(self.right_basket_pos, axis=0),
                np.expand_dims([self.court_length / 2, 0], axis=0),
                np.expand_dims([self.court_length, self.court_width], axis=0)
            ], axis=0)
        return obs

    def reset(self):
        """ random init positions in the right half court
        1. init offensive team randomlu
        2. add defensive team next to each offensive player in the basket side.
        """
        if self.init_mode == INIT_LOOKUP['INPUT']:
            assert self.init_positions is not None
            assert self.init_ball_handler_idx is not None
            self.states.reset(
                self.init_positions, self.init_ball_handler_idx, buffer_size=self.buffer_size)
        elif self.init_mode == INIT_LOOKUP['DATASET']:
            data = np.load('bball_strategies/data/FrameRate5.npy')
            ep_idx = np.floor(self.np_random_generator.uniform(
                low=0.0, high=data.shape[0])).astype(np.int)
            ball_pos = data[ep_idx, 0, 0, 0:2]
            off_positions = data[ep_idx, 0, 1:6, 0:2]
            def_positions = data[ep_idx, 0, 6:11, 0:2]
            off2ball_vec = off_positions - ball_pos
            ball_handler_idx = np.argmin(length(off2ball_vec, axis=1))
            positions = np.array(
                [off_positions[ball_handler_idx], off_positions, def_positions])
            vels = np.array([np.zeros_like(ball_pos, dtype=np.float32), np.zeros_like(
                off_positions, dtype=np.float32), np.zeros_like(def_positions, dtype=np.float32)])
            self.states.reset(positions, ball_handler_idx,
                              buffer_size=self.buffer_size)
        else:
            if self.init_mode == INIT_LOOKUP['DEFAULT']:
                off_positions = np.array([
                    [80, 40],
                    [70, 35],
                    [60, 25],
                    [70, 15],
                    [80, 10]
                ], dtype=np.float32)
                ball_handler_idx = 2
            else:
                off_positions = self.np_random_generator.uniform(
                    low=[self.court_length // 2, 0], high=[self.court_length, self.court_width], size=[5, 2])
                ball_handler_idx = np.floor(self.np_random_generator.uniform(
                    low=0.0, high=5.0)).astype(np.int)

            def_positions = np.array(
                off_positions, copy=True, dtype=np.float32)
            vec = self.right_basket_pos - off_positions
            vec_length = length(vec, axis=1)
            # vec_length = np.sqrt(np.sum(vec * vec, axis=1))
            u_vec = vec / np.stack([vec_length, vec_length], axis=1)
            def_positions = def_positions + u_vec * self.screen_radius * 2
            ball_pos = np.array(
                off_positions[ball_handler_idx, :], copy=True, dtype=np.float32)
            positions = np.array([ball_pos, off_positions, def_positions])
            self.states.reset(positions, ball_handler_idx,
                              buffer_size=self.buffer_size)

        return self._get_obs()

    # def render(self, mode='human', close=False):
        # if close:
        #     if self.viewer is not None:
        #         self.viewer.close()
        #         self.viewer = None
        #     return
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
        if not self.if_vis_trajectory:
            if self.viewer is None:
                self.viewer = rendering.Viewer(940+100, 500+100)
                self.viewer.set_bounds(0-5, 94+5, 0-5, 50+5)  # feet
                # background img
                fname = path.join(path.dirname(__file__), "fullcourt.png")
                img = rendering.Image(fname, 94, 50)
                imgtrans = rendering.Transform(translation=(47.0, 25.0))
                img.add_attr(imgtrans)
                self.viewer.add_geom(img)
                # defensive players
                for _ in range(5):
                    def_player = rendering.make_circle(radius=2.)
                    def_player.set_color(0, 0, 1)
                    def_trans = rendering.Transform()
                    self.def_pl_transforms.append(def_trans)
                    def_player.add_attr(def_trans)
                    self.viewer.add_geom(def_player)
                    if self.if_vis_visual_aid:
                        def_player_screen = rendering.make_circle(
                            radius=self.screen_radius, filled=False)
                        def_player_wingspan = rendering.make_circle(
                            radius=self.wingspan_radius, filled=False)
                        def_player_screen.set_color(0, 0, 0.75)
                        def_player_wingspan.set_color(0.5, 0.5, 0.5)
                        def_player_screen.add_attr(def_trans)
                        def_player_wingspan.add_attr(def_trans)
                        self.viewer.add_geom(def_player_screen)
                        self.viewer.add_geom(def_player_wingspan)
                # offensive players
                for _ in range(5):
                    off_player = rendering.make_circle(radius=2.)
                    off_player.set_color(1, 0, 0)
                    off_trans = rendering.Transform()
                    self.off_pl_transforms.append(off_trans)
                    off_player.add_attr(off_trans)
                    self.viewer.add_geom(off_player)
                    if self.if_vis_visual_aid:
                        off_player_screen = rendering.make_circle(
                            radius=self.screen_radius, filled=False)
                        off_player_wingspan = rendering.make_circle(
                            radius=self.wingspan_radius, filled=False)
                        off_stolen_range = rendering.make_circle(
                            radius=self.stolen_radius, filled=False)
                        off_player_screen.set_color(0.75, 0, 0)
                        off_player_wingspan.set_color(0.5, 0.5, 0.5)
                        off_stolen_range.set_color(0, 0, 0.75)
                        off_player_screen.add_attr(off_trans)
                        off_player_wingspan.add_attr(off_trans)
                        off_stolen_range.add_attr(off_trans)
                        self.viewer.add_geom(off_player_screen)
                        self.viewer.add_geom(off_player_wingspan)
                        self.viewer.add_geom(off_stolen_range)
                # ball
                ball = rendering.make_circle(radius=1.)
                ball.set_color(0, 1, 0)
                ball_trans = rendering.Transform()
                self.ball_transform = ball_trans
                ball.add_attr(ball_trans)
                self.viewer.add_geom(ball)

            ### set translations ###
            # defensive players
            for trans, pos in zip(self.def_pl_transforms, self.states.defense_positions):
                trans.set_translation(pos[0], pos[1])
            # offensive players
            for trans, pos in zip(self.off_pl_transforms, self.states.offense_positions):
                trans.set_translation(pos[0], pos[1])
            # ball
            ball_pos = self.states.ball_position
            self.ball_transform.set_translation(ball_pos[0], ball_pos[1])

        if self.if_vis_trajectory:
            if self.viewer is None:
                self.viewer = rendering.Viewer(940, 500)
                self.viewer.set_bounds(0, 94, 0, 50)  # feet
                # background img
                fname = path.join(path.dirname(__file__), "fullcourt.png")
                img = rendering.Image(fname, 94, 50)
                imgtrans = rendering.Transform(translation=(47.0, 25.0))
                img.add_attr(imgtrans)
                self.viewer.add_geom(img)
            # defensive players
            for i in range(5):
                def_player = rendering.make_circle(radius=2.)
                def_player.set_color(0, 0, 1)
                def_trans = rendering.Transform()
                pos = self.states.defense_positions[i]
                def_trans.set_translation(pos[0], pos[1])
                def_player.add_attr(def_trans)
                self.viewer.add_geom(def_player)
                if self.if_vis_visual_aid:
                    def_player_screen = rendering.make_circle(
                        radius=self.screen_radius, filled=False)
                    def_player_wingspan = rendering.make_circle(
                        radius=self.wingspan_radius, filled=False)
                    def_player_screen.set_color(0, 0, 0.75)
                    def_player_wingspan.set_color(0.5, 0.5, 0.5)
                    def_player_screen.add_attr(def_trans)
                    def_player_wingspan.add_attr(def_trans)
                    self.viewer.add_geom(def_player_screen)
                    self.viewer.add_geom(def_player_wingspan)
            # offensive players
            for i in range(5):
                off_player = rendering.make_circle(radius=2.)
                off_player.set_color(1, 0, 0)
                off_trans = rendering.Transform()
                pos = self.states.offense_positions[i]
                off_trans.set_translation(pos[0], pos[1])
                off_player.add_attr(off_trans)
                self.viewer.add_geom(off_player)
                if self.if_vis_visual_aid:
                    off_player_screen = rendering.make_circle(
                        radius=self.screen_radius, filled=False)
                    off_player_wingspan = rendering.make_circle(
                        radius=self.wingspan_radius, filled=False)
                    off_stolen_range = rendering.make_circle(
                        radius=self.stolen_radius, filled=False)
                    off_player_screen.set_color(0.75, 0, 0)
                    off_player_wingspan.set_color(0.5, 0.5, 0.5)
                    off_stolen_range.set_color(0, 0, 0.75)
                    off_player_screen.add_attr(off_trans)
                    off_player_wingspan.add_attr(off_trans)
                    off_stolen_range.add_attr(off_trans)
                    self.viewer.add_geom(off_player_screen)
                    self.viewer.add_geom(off_player_wingspan)
                    self.viewer.add_geom(off_stolen_range)
            # ball
            ball = rendering.make_circle(radius=1.)
            ball.set_color(0, 1, 0)
            ball_trans = rendering.Transform()
            ball_pos = self.states.ball_position
            ball_trans.set_translation(ball_pos[0], ball_pos[1])
            ball.add_attr(ball_trans)
            self.viewer.add_geom(ball)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def seed(self, seed=None):
        # seeding use very strong random seed generated by os
        self.np_random_generator, seed = seeding.np_random(seed)
        return [seed]

    def _set_action_space(self):
        """
        Return
        ------
        Tuple(Discrete(3), Box(), Box(5, 2), Box(5, 2))
        """
        return tools.ActTuple((
            spaces.Discrete(3),  # offensive decision
            # ball theta
            spaces.Box(
                low=-np.pi, high=np.pi, shape=(), dtype=np.float32
            ),
            # offense player DASH(power, direction)
            spaces.Box(
                low=np.array([[0, -np.pi]
                              for _ in range(5)], dtype=np.float32),
                high=np.array([[self.pl_max_power, np.pi]
                               for _ in range(5)], dtype=np.float32),
                dtype=np.float32
            ),
            # defense player DASH(power, direction)
            spaces.Box(
                low=np.array([[0, -np.pi]
                              for _ in range(5)], dtype=np.float32),
                high=np.array([[self.pl_max_power, np.pi]
                               for _ in range(5)], dtype=np.float32),
                dtype=np.float32
            )
        ))

    def _set_observation_space(self):
        """
        ## shape
        5 : 5 frames
        14 : ball(1) + offense(5) + defense(5) + basket(1) + ball_boundry(2)
        2 : x and y positions
        """
        # # Tuple(Box(2,), Box(5, 2), Box(5, 2))
        # return tools.ObsTuple((
        #     # ball position
        #     spaces.Box(low=np.array([self.court_length // 2, 0]),
        #                high=np.array([self.court_length, self.court_width])),
        #     # offense player positions
        #     spaces.Box(low=np.array([[0, 0] for _ in range(5)]),
        #                high=np.array([[self.court_length, self.court_width] for _ in range(5)])),
        #     # defense player positions
        #     spaces.Box(low=np.array([[0, 0] for _ in range(5)]),
        #                high=np.array([[self.court_length, self.court_width] for _ in range(5)]))
        # ))

        # boundries are defined in self.states._clip_state(pos)
        # return spaces.Box(low=-np.inf, high=np.inf, shape=(10, 14, 2), dtype=np.float32)
        # return spaces.Box(low=-np.inf, high=np.inf, shape=(5, 14, 2), dtype=np.float32)

        low_ = np.array([self.states.x_low_bound, self.states.y_low_bound]) * \
            np.ones(shape=[self.buffer_size, 14, 2])
        high_ = np.array([self.states.x_high_bound, self.states.y_high_bound]) * \
            np.ones(shape=[self.buffer_size, 14, 2])
        return spaces.Box(low=low_, high=high_, dtype=np.float32)

    def _update_player_state(self, pl_dash, vels, state_idx):
        """ Update the player's movement following the physics limitation predefined
        Inputs
        ------
        pl_dash : float, shape=[5,2]
            the force of each player
        """
        # 1. update the vels by player's power (with max speed limit = self.pl_max_speed)
        # 2. check if any collisions might happen among one player to five opponents.
        # 3. if any collisions, choose the closest one, calculate the discounted velocity, then update player
        # 4. if none collisions, update with velocity

        # 1. update the vels by player's power (with max speed limit = self.pl_max_speed)
        # update player state
        assert pl_dash.shape == (5, 2)
        # decomposing into power and direction
        pl_power = np.clip(
            pl_dash[:, DASH_LOOKUP['POWER']], 0, self.pl_max_power)
        pl_acc_dir = np.clip(
            pl_dash[:, DASH_LOOKUP['DIRECTION']], -np.pi, np.pi)
        pl_acc_dir_vec = np.stack(
            [np.cos(pl_acc_dir), np.sin(pl_acc_dir)], axis=1)
        pl_acc_vec = np.stack(
            [pl_acc_dir_vec[:, 0] * pl_power, pl_acc_dir_vec[:, 1] * pl_power], axis=1)
        assert pl_acc_vec.shape == vels[state_idx].shape
        pl_vels = np.add(vels[state_idx], pl_acc_vec)
        pl_speed = length(pl_vels, axis=1)
        # can't not exceed the limits
        indices = np.argwhere(pl_speed >= self.pl_max_speed)
        pl_vels[indices] = pl_vels[indices] / \
            np.stack([pl_speed[indices], pl_speed[indices]],
                     axis=-1) * self.pl_max_speed
        pl_speed = length(pl_vels, axis=1)
        # current moving direction (unit vector)
        pl_vels_dir_vec = np.empty_like(pl_vels, dtype=np.float32)
        for i in range(pl_vels_dir_vec.shape[0]):
            # avoid divide zero
            pl_vels_dir_vec[i] = [0, 0] if pl_speed[i] == 0 else [
                pl_vels[i, 0] / pl_speed[i], pl_vels[i, 1] / pl_speed[i]]

        # 2. check if any collisions might happen among one player to five opponents.
        if state_idx == STATE_LOOKUP['DEFENSE']:
            team_id = FLAG_LOOKUP['DEFENSE']
            opp_idx = STATE_LOOKUP['OFFENSE']
        elif state_idx == STATE_LOOKUP['OFFENSE']:
            team_id = FLAG_LOOKUP['OFFENSE']
            opp_idx = STATE_LOOKUP['DEFENSE']
        opp_positions = self.states.positions[opp_idx]
        for pl_idx, (pl_pos, next_pl_pos, pl_vel) in enumerate(zip(self.states.positions[state_idx], self.states.positions[state_idx] + pl_vels, pl_vels)):
            zone_idx, mode = self._find_zone_idx(
                next_pl_pos, pl_pos, pl_vel, opp_positions, self.screen_radius, team_id)
        # 3. if any collisions, choose the closest one, calculate the discounted velocity, then update player
            if zone_idx is not None and mode != MODE_LOOKUP['LEAVE-CATCH']:
                logger.debug(
                    '[COLLISION] mode {}, {}-{} colide into {}-{}'.format(mode, state_idx, pl_idx, opp_idx, zone_idx))
                if mode == MODE_LOOKUP['IN'] or mode == MODE_LOOKUP['IN-THEN-OUT']:
                    # if 'IN' or 'IN-THEN-OUT', use collision_speed
                    collision_speed = self.pl_collision_speed
                    collision_speed = collision_speed if pl_speed[
                        pl_idx] >= collision_speed else pl_speed[pl_idx]
                    pl_collision_vel = collision_speed * \
                        pl_vels_dir_vec[pl_idx]
                elif mode == MODE_LOOKUP['COME-THEN-IN']:
                    # if 'COME-THEN-IN', combine player speed when 'COME' and collision_speed when 'IN'
                    # a. length between player and opponent
                    pl2zone_vec = opp_positions[zone_idx] - pl_pos
                    pl2zone_length = length(pl2zone_vec, axis=0)
                    # b. length between player and opponent on velocity direction = a dot dir
                    pl2zone_ondir_length = np.dot(
                        pl2zone_vec, pl_vels_dir_vec[pl_idx])
                    # c. legnth from player to circle edge = b - sqrt(rr-(aa-bb))
                    # zone to end pos
                    temp = pl2zone_length**2 - pl2zone_ondir_length**2
                    # edge to end pos
                    temp2 = self.screen_radius**2 - temp
                    if temp < 1e-5:
                        pl2edge_length = abs(
                            pl2zone_length - self.screen_radius)
                    else:
                        pl2edge_length = pl2zone_ondir_length - np.sqrt(temp2)
                    # d. final velocity = c * direction + sqrt(rr-(aa-bb)) * collision_speed * direction
                    collision_speed = self.pl_collision_speed if pl_speed[
                        pl_idx] >= self.pl_collision_speed else pl_speed[pl_idx]
                    pl_collision_vel = pl2edge_length * pl_vels_dir_vec[pl_idx] + \
                        (1 - pl2edge_length / pl_speed[pl_idx]) * \
                        collision_speed * pl_vels_dir_vec[pl_idx]
                next_pl_pos = pl_pos + pl_collision_vel
                pl_vel = 0.0  # if collision, velocity = 0
        # 4. if none collisions, update with velocity
            self.states.update_player(state_idx, pl_idx, next_pl_pos, pl_vel)

    def _find_zone_idx(self, next_pos, pos, vel, zone_positions, zone_radius, team_id):
        shortest_dists = np.empty(
            shape=(len(zone_positions),), dtype=np.float32)
        modes = np.empty(shape=(len(zone_positions),), dtype=np.float32)
        next_pos2zone_vecs = zone_positions - next_pos
        pos2zone_vecs = zone_positions - pos
        for i, (next_vec, vec) in enumerate(zip(next_pos2zone_vecs, pos2zone_vecs)):
            next_dotvalue = np.dot(next_vec, vel)
            dotvalue = np.dot(vec, vel)
            vec2zone_dist = length(vec, axis=0)
            if vec2zone_dist <= zone_radius and dotvalue > 0.0:                    
                modes[i] = MODE_LOOKUP['IN']
                temp = np.inner(next_vec, np.multiply(-1, vel)
                                ) / length(vel, axis=0)
                shortest_dists[i] = 0.0 if abs(length(
                    next_vec, axis=0)**2 - temp**2) < 1e-5 else np.sqrt(length(next_vec, axis=0)**2 - temp**2)
            elif next_dotvalue <= 0.0 and dotvalue <= 0.0:  # out or leave
                if team_id == FLAG_LOOKUP['OFFENSE']:
                    if length(vec, axis=0) < 1e-8:
                        modes[i] = MODE_LOOKUP['LEAVE']
                        shortest_dists[i] = sys.float_info.max
                    else:
                        modes[i] = MODE_LOOKUP['LEAVE-CATCH']
                        shortest_dists[i] = length(vec, axis=0)
                elif team_id == FLAG_LOOKUP['DEFENSE']:
                    modes[i] = MODE_LOOKUP['LEAVE']
                    shortest_dists[i] = sys.float_info.max
            elif next_dotvalue > 0.0 and dotvalue > 0.0:  # in or come
                next2zone_dist = length(next_vec, axis=0)
                if next2zone_dist <= zone_radius:
                    modes[i] = MODE_LOOKUP['COME-THEN-IN']
                    shortest_dists[i] = next2zone_dist
                else:
                    modes[i] = MODE_LOOKUP['COME']
                    shortest_dists[i] = sys.float_info.max
            elif next_dotvalue <= 0.0 and dotvalue > 0.0:  # in then out
                modes[i] = MODE_LOOKUP['IN-THEN-OUT']
                temp = np.inner(next_vec, np.multiply(-1, vel)
                                ) / length(vel, axis=0)
                shortest_dists[i] = 0.0 if abs(length(
                    next_vec, axis=0)**2 - temp**2) < 1e-5 else np.sqrt(length(next_vec, axis=0)**2 - temp**2)
        candidates = np.argwhere(
            shortest_dists <= zone_radius).reshape([-1])
        zone_idx = None
        if len(candidates) != 0:
            if len(candidates) > 1:
                zone_idx = candidates[
                    np.argmin(length(pos2zone_vecs[candidates], axis=1))]
            else:
                zone_idx = candidates[0]
        return zone_idx, modes[zone_idx]

    def _update_ball_state(self, decision, ball_pass_dir):
        """
        Inputs
        ------
        decision : int
            offensive team's decision, including SHOOT, NOOP, and PASS.
        """
        if self.states.is_passing or decision == DESICION_LOOKUP['PASS']:
            if decision == DESICION_LOOKUP['PASS']:
                if self.states.is_passing:
                    # maybe return negative reward!?
                    logger.debug('[BALL] You cannot do PASS decision while ball is passing')
                else:
                    ball_pass_dir = np.clip(
                        ball_pass_dir, -np.pi, np.pi)
                    new_vel = [self.ball_passing_speed * np.cos(ball_pass_dir),
                            self.ball_passing_speed * np.sin(ball_pass_dir)]
                    self.states.update_ball(self.states.ball_position,
                                            None, True, new_vel)
                    logger.debug('[BALL] Start Flying')
            
            # check if ball caught/stolen
            # Prerequisites:
            # 1. if any defenders is close enough to offender (depend on stolen_radius)
            # 2. defender is closer to ball init position than offender
            # 3. if any in (2), check is any defender able to fetch ball (depend on defender's wingspan_radius), then go (5)
            # 4. if none in (2), check is any offender able to fetch ball (depend on offender's wingspan_radius), then go (5)
            # 5. if any candidates, choose the best catcher among them
            # 6. if none candidates, ball keep flying
            ball_pos = self.states.ball_position
            next_ball_pos = self.states.ball_position + self.states.ball_vel
            off2oldball_vecs = self.states.offense_positions - ball_pos
            def2oldball_vecs = self.states.defense_positions - ball_pos

            # 1. if any defenders is close enough to offender (depend on stolen_radius)
            # 2. defender is closer to ball init position than offender
            candidates = []
            for key, value in self.states.off_def_closest_map.items():
                if value['distance'] <= self.stolen_radius:
                    if length(def2oldball_vecs[value['idx']], axis=0) <= length(off2oldball_vecs[key], axis=0):
                        candidates.append(value['idx'])
            # TODO skip
            # 3. if any in (2), check is any defender able to fetch ball (depend on defender's wingspan_radius), then go (5)
            def_zone_idx = None
            # if len(candidates) != 0:
            #     def_zone_idx, _ = self._find_zone_idx(
            #         next_ball_pos, ball_pos, self.states.ball_vel, self.states.defense_positions[candidates], self.wingspan_radius, FLAG_LOOKUP['DEFENSE'])
            # # 5. if any candidates, choose the best catcher among them
            #     if def_zone_idx is not None:
            #         # assign catcher pos to ball pos
            #         self.states.update_ball(self.states.defense_positions[candidates][def_zone_idx],
            #                                 None, False, [0, 0])
            #         self.states.update_status(
            #             done=True, status=STATUS_LOOKUP['CAPTURED'])
            #         return
            # 4. if none in (2), check is any offender able to fetch ball (depend on offender's wingspan_radius), then go (5)
            off_zone_idx = None
            if len(candidates) == 0 or (def_zone_idx is None):
                off_zone_idx, _ = self._find_zone_idx(
                    next_ball_pos, ball_pos, self.states.ball_vel, self.states.offense_positions, self.wingspan_radius, FLAG_LOOKUP['OFFENSE'])
            # 5. if any candidates, choose the best catcher among them
                if off_zone_idx is not None:
                    # assign catcher pos to ball pos
                    self.states.update_ball(self.states.offense_positions[off_zone_idx],
                                            off_zone_idx, False, [0, 0])
                    self.states.update_status(
                        done=False, status=STATUS_LOOKUP['CATCH'])
                    return
            # 6. if none candidates, ball keep flying
            if def_zone_idx is None and off_zone_idx is None:
                self.states.update_ball(self.states.ball_position + self.states.ball_vel,
                                        None, True, self.states.ball_vel)
                logger.debug('[BALL] Flying')
                return
        elif decision == DESICION_LOOKUP['SHOOT'] or decision == DESICION_LOOKUP['NO_OP']:
            # dribble, assign ball handler's position to ball
            self.states.update_ball(self.states.offense_positions[self.states.ball_handler_idx],
                                    self.states.ball_handler_idx, False, [0, 0])
            return

    def _calculate_reward(self):
        # 1. find distances between all defenders to ball handler
        def2ball_vecs = self.states.ball_position - self.states.defense_positions
        def2ball_lens = length(def2ball_vecs, axis=1)
        # NOTE if too close, get -1
        def2ball_lens[np.where(def2ball_lens < 10.0)] = -1

        # 2. update distance to max if any offensive player in between
        off2ball_vecs = self.states.ball_position - self.states.offense_positions
        for i, (def2ball_vec, _) in enumerate(zip(def2ball_vecs, def2ball_lens)):
            off2def_vecs = self.states.defense_positions[i] - \
                self.states.offense_positions
            off_dot_def = np.inner(off2ball_vecs, def2ball_vec)
            off_dot_ball = np.inner(off2def_vecs, -1 * def2ball_vec)
            if_inbetween = np.logical_and(off_dot_def > 0, off_dot_ball > 0)
            indices = np.argwhere(if_inbetween)[:, 0]
            off2vec_lens = np.sqrt(abs(
                length(off2ball_vecs[indices], axis=1)**2 - (off_dot_def[indices] / length(def2ball_vec, axis=0))**2))
            screen_indices = np.argwhere(off2vec_lens < self.screen_radius)
            if len(screen_indices) > 0:
                def2ball_lens[i] = self.max_reward
        # 3. calculate the weighted sum of length and angle value
        ball2cloesetdef_vecs = self.states.defense_positions - \
            self.states.ball_position
        ball2basket_vec = self.right_basket_pos - self.states.ball_position
        ball2basket_len = length(ball2basket_vec, axis=0)
        ball_dot_defs = np.inner(ball2cloesetdef_vecs, ball2basket_vec)
        def2ball_lens[np.argwhere(ball_dot_defs <= 0)] = self.max_reward
        # avoid divide zero
        angle_values = np.empty(shape=(5,), dtype=np.float32)
        len_temp = length(ball2cloesetdef_vecs, axis=1) * \
            length(ball2basket_vec, axis=0)
        for i in range(5):
            if len_temp[i] == 0.0:
                angle_values[i] = 0.0
            else:
                angle_values[i] = np.arccos(ball_dot_defs[i] / len_temp[i])
        penalty = (self.three_point_distance -
                   ball2basket_len) if ball2basket_len > self.three_point_distance else 0.0
        rewards = self.len2pi_weight * def2ball_lens + \
            angle_values + self.len2pi_weight * penalty
        # 4. find the best defender to ball handler
        reward = np.amin(rewards)

        return reward


class States(object):
    """ modulize all states into class """

    def __init__(self):
        """
        """
        # ball and players [ball, offense, defense]
        self.turn = None
        self.positions = None
        self.buffer_positions = None
        self.vels = None
        self.done = None
        self.status = None
        self.steps = None
        # self.__dirs = None
        # ball state
        self.is_passing = None
        self.ball_handler_idx = None
        self.off_def_closest_map = None
        # court information
        self.clip_padding = 5.0
        self.x_high_bound = 94.0 + self.clip_padding
        self.x_low_bound = 94.0/2.0 - self.clip_padding
        self.y_high_bound = 50.0 + self.clip_padding
        self.y_low_bound = 0 - self.clip_padding

    def reset(self, positions, ball_handler_idx, buffer_size=5):
        self.turn = FLAG_LOOKUP['OFFENSE']
        self.positions = positions
        # fill all with first frames
        self.buffer_positions = np.tile(positions, [buffer_size, 1])
        self.vels = np.array([np.zeros_like(positions[0], dtype=np.float32), np.zeros_like(
            positions[1], dtype=np.float32), np.zeros_like(positions[2], dtype=np.float32)])
        self.done = False
        self.status = None
        self.steps = 0
        # self.__dirs = None
        # ball state
        self.ball_handler_idx = ball_handler_idx
        self.is_passing = False
        self.off_def_closest_map = dict()
        self.update_closest_map()

    def update_ball(self, pos, ball_handler_idx, is_passing, velocity):
        # clip
        pos = self._clip_state(pos)
        # update
        self.ball_handler_idx = ball_handler_idx
        self.is_passing = is_passing
        self.positions[STATE_LOOKUP['BALL']] = pos
        self.vels[STATE_LOOKUP['BALL']] = np.array(velocity, dtype=np.float32)

    def update_player(self, team_id, pl_idx, position, vel):
        # clip
        position = self._clip_state(position)
        # update
        self.positions[team_id][pl_idx] = position
        self.vels[team_id][pl_idx] = vel

    def take_turn(self):
        """
        self.buffer_positions : 
        t : index of frame
        if offensive turn : [ball,off,def] 5 frames as obs -> [t-4, t-4, t-5],[t-3, t-3, t-4],[t-2, t-2, t-3],[t-1, t-1, t-2],[t-0, t-0, t-1]
        if defensive turn : [ball,off,def] 5 frames as obs -> [t-4, t-4, t-4],[t-3, t-3, t-3],[t-2, t-2, t-2],[t-1, t-1, t-1],[t-0, t-0, t-0]
        """
        def update_buffer(team_idx):
            for last_pos, pos in zip(self.buffer_positions[:-1], self.buffer_positions[1:]):
                if team_idx==STATE_LOOKUP['OFFENSE']:
                    last_pos[STATE_LOOKUP['BALL']] = copy.deepcopy(pos[STATE_LOOKUP['BALL']])
                last_pos[team_idx] = copy.deepcopy(pos[team_idx])
            if team_idx==STATE_LOOKUP['OFFENSE']:
                self.buffer_positions[-1][STATE_LOOKUP['BALL']] = copy.deepcopy(self.positions[STATE_LOOKUP['BALL']])
            self.buffer_positions[-1][team_idx] = copy.deepcopy(self.positions[team_idx])
        team_idx = STATE_LOOKUP['OFFENSE'] if self.turn == FLAG_LOOKUP['OFFENSE'] else STATE_LOOKUP['DEFENSE']
        update_buffer(team_idx)

        self.turn = FLAG_LOOKUP['DEFENSE'] if self.turn == FLAG_LOOKUP['OFFENSE'] else FLAG_LOOKUP['OFFENSE']
        self.steps = self.steps + 1
        self.update_closest_map()
        self.status = None

    def update_status(self, done, status):
        self.done = done
        self.status = status

    def update_closest_map(self):
        for idx, off_pos in enumerate(self.offense_positions):
            distances = distance(self.defense_positions, off_pos)
            temp_idx = np.argmin(distances)
            self.off_def_closest_map[idx] = {
                'idx': temp_idx, 'distance': distances[temp_idx]}

    def _clip_state(self, pos):
        pos[0] = self.x_low_bound if pos[0] < self.x_low_bound else pos[0]
        pos[0] = self.x_high_bound if pos[0] >= self.x_high_bound else pos[0]
        pos[1] = self.y_low_bound if pos[1] < self.y_low_bound else pos[1]
        pos[1] = self.y_high_bound if pos[1] >= self.y_high_bound else pos[1]
        return pos

    @property
    def ball_position(self):
        return self.positions[STATE_LOOKUP['BALL']]

    @property
    def offense_positions(self):
        return self.positions[STATE_LOOKUP['OFFENSE']]

    @property
    def defense_positions(self):
        return self.positions[STATE_LOOKUP['DEFENSE']]

    @property
    def ball_vel(self):
        return self.vels[STATE_LOOKUP['BALL']]


# Termination Conditions:
STATUS_LOOKUP = {
    'SHOOT': 0,  # A shoot decision is made
    'CAPTURED': 1,  # A defender gets possession of the ball
    'OOB': 2,  # The ball is Out Of Bounds
    'OOT': 3,  # Out Of Time by a max time limitation
    'CATCH': 4  # Succefully Pass Ball
}


MODE_LOOKUP = {
    'LEAVE': 0,
    'COME': 1,
    'IN': 2,
    'COME-THEN-IN': 3,
    'IN-THEN-OUT': 4,
    'LEAVE-CATCH': 5
}

DASH_LOOKUP = {
    'POWER': 0,
    'DIRECTION': 1
}

STATE_LOOKUP = {
    # Tuple(Box(2,), Box(5, 2), Box(5, 2))
    'BALL': 0,
    'OFFENSE': 1,
    'DEFENSE': 2
}

ACTION_LOOKUP = {
    # Tuple(Discrete(3), Box(), Box(5, 2), Box(5, 2))
    'DECISION': 0,
    'BALL': 1,
    'OFF_DASH': 2,
    'DEF_DASH': 3
}

FLAG_LOOKUP = {
    'OFFENSE': 0,
    'DEFENSE': 1
}

DESICION_LOOKUP = {
    # Discrete(3)
    'SHOOT': 0,
    'PASS': 1,
    'NO_OP': 2
}

INIT_LOOKUP = {
    'DEFAULT': 0,
    'DATASET': 1,
    'INPUT': 2
}
