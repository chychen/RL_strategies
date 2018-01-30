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
from gym.envs.classic_control import rendering
"""
# TODO
# 把一些環境資訊物件化
# check what is the screen range?
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
        self.if_init_by_default = False
        self.if_init_by_dataset = False
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
        self.time_limit = 24 * FPS * 2
        self.left_basket_pos = [0 + 5.25, 25]
        self.right_basket_pos = [94 - 5.25, 25]
        # this is the distance between two players' center position
        self.screen_radius = 2.0 * 2
        self.wingspan_radius = 3.5
        self.stolen_radius = 5.0
        # physics limitations TODO per frame
        self.ball_passing_speed = 30.0 / FPS
        self.pl_max_speed = 38.9379818754 / FPS
        # cost at least one second to cross over the opponent
        self.pl_collision_speed = self.screen_radius / FPS
        self.pl_max_power = 24.5810950984 / FPS
        # Env information
        self.states = States()
        self.off_def_closest_map = None  # dict()

        # must define properties
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

        # random seed
        self._seed()

    def _step(self, action):
        """
        Returns
        -------
        observation (object) : agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean) : whether the episode has ended, in which case further step() calls will return undefined results
        info (dict) : contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        decision = action[ACTION_LOOKUP['DECISION']]
        ball_pass_dir = action[ACTION_LOOKUP['BALL']]
        pl_dash = action[ACTION_LOOKUP['DASH']]
        if self.states.turn == FLAG_LOOPUP['OFFENSE']:
            logger.info('[TURN] OFFENSE')
            self._update_player_state(
                pl_dash, self.states.vels, STATE_LOOKUP['OFFENSE'])
            # update ball state
            self._update_ball_state(decision, ball_pass_dir)
        elif self.states.turn == FLAG_LOOPUP['DEFENSE']:
            logger.info('[TURN] DEFENSE')
            self._update_player_state(
                pl_dash, self.states.vels, STATE_LOOKUP['DEFENSE'])

        # check if meets termination condition
        if decision == DESICION_LOOKUP['SHOOT']:
            # self.states.game_over(REASON_LOOKUP['SHOOT'])
            pass
        # OOB
        if self.states.ball_position[0] >= self.court_length or self.states.ball_position[0] < 0.0 or self.states.ball_position[1] >= self.court_width or self.states.ball_position[1] < 0.0:
            # self.states.game_over(REASON_LOOKUP['OOB'])
            pass
        # OOT
        if self.states.steps >= self.time_limit:
            # self.states.game_over(REASON_LOOKUP['OOT'])
            pass
        # termination conditions
        if self.states.done:
            if self.states.reason == REASON_LOOKUP['SHOOT']:
                logger.info('[GAME OVER], A shoot decision is made')
                reward = self._calculate_reward()
                pass
            elif self.states.reason == REASON_LOOKUP['CAPTURED']:
                logger.info(
                    '[GAME OVER], A defender gets possession of the ball')
                pass
            elif self.states.reason == REASON_LOOKUP['OOB']:
                logger.info('[GAME OVER], The ball is out of bounds.')
                pass
            elif self.states.reason == REASON_LOOKUP['OOT']:
                logger.info(
                    '[GAME OVER], Max time limit for the episode is reached')
                pass

        # update env information
        self.update_closest_map()

        return self.states.positions, 0.0, self.states.done, dict()

    def _reset(self):
        """ random init positions in the right half court
        1. init offensive team randomlu
        2. add defensive team next to each offensive player in the basket side.
        """
        if self.if_init_by_dataset:
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
            vels = np.array([np.zeros_like(ball_pos), np.zeros_like(
                off_positions), np.zeros_like(def_positions)])
            self.states.reset(positions, vels, ball_handler_idx)
        else:
            if self.if_init_by_default:
                off_positions = np.array([
                    [80, 40],
                    [70, 35],
                    [60, 25],
                    [70, 15],
                    [80, 10]
                ], dtype=np.float)
                ball_handler_idx = 2
            else:
                off_positions = self.np_random_generator.uniform(
                    low=[self.court_length // 2, 0], high=[self.court_length, self.court_width], size=[5, 2])
                ball_handler_idx = np.floor(self.np_random_generator.uniform(
                    low=0.0, high=5.0)).astype(np.int)

            def_positions = np.array(off_positions, copy=True)
            vec = self.right_basket_pos - off_positions
            vec_length = length(vec, axis=1)
            # vec_length = np.sqrt(np.sum(vec * vec, axis=1))
            u_vec = vec / np.stack([vec_length, vec_length], axis=1)
            def_positions = def_positions + u_vec * self.screen_radius * 2
            ball_pos = np.array(off_positions[ball_handler_idx, :], copy=True)
            positions = np.array([ball_pos, off_positions, def_positions])
            vels = np.array([np.zeros_like(ball_pos), np.zeros_like(
                off_positions), np.zeros_like(def_positions)])
            self.states.reset(positions, vels, ball_handler_idx)
        
        self.off_def_closest_map = dict()
        self.update_closest_map()

        return self.states.positions

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if not self.if_vis_trajectory:
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
        Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
        """
        return spaces.Tuple((
            spaces.Discrete(3),  # offensive decision
            # ball theta
            spaces.Box(
                low=-np.pi, high=np.pi, shape=()
            ),
            # offense player DASH(power, direction)
            spaces.Box(
                low=np.array([[0, -np.pi] for _ in range(5)]),
                high=np.array([[self.pl_max_power, np.pi] for _ in range(5)])
            ),
            # defense player DASH(power, direction)
            spaces.Box(
                low=np.array([[0, -np.pi] for _ in range(5)]),
                high=np.array([[self.pl_max_power, np.pi] for _ in range(5)])
            )
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
        indices = np.argwhere(pl_speed > self.pl_max_speed)
        pl_vels[indices] = pl_vels[indices] / \
            np.stack([pl_speed[indices], pl_speed[indices]],
                     axis=-1) * self.pl_max_speed
        pl_speed = length(pl_vels, axis=1)
        # current moving direction (unit vector)
        pl_vels_dir_vec = np.stack(
            [pl_vels[:, 0] / pl_speed, pl_vels[:, 1] / pl_speed], axis=1)

        # 2. check if any collisions might happen among one player to five opponents.
        if state_idx == STATE_LOOKUP['DEFENSE']:
            opp_idx = STATE_LOOKUP['OFFENSE']
        elif state_idx == STATE_LOOKUP['OFFENSE']:
            opp_idx = STATE_LOOKUP['DEFENSE']
        opp_positions = self.states.positions[opp_idx]
        for pl_idx, (pl_pos, next_pl_pos, pl_vel) in enumerate(zip(self.states.positions[state_idx], self.states.positions[state_idx] + pl_vels, pl_vels)):
            zone_idx, mode = self._find_zone_idx(
                next_pl_pos, pl_pos, pl_vel, opp_positions, self.screen_radius)
        # 3. if any collisions, choose the closest one, calculate the discounted velocity, then update player
            if zone_idx is not None:
                logger.info(
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
        self.states.take_turn()

    def _find_zone_idx(self, next_pos, pos, vel, zone_positions, zone_radius):
        shortest_dists = np.empty(shape=(len(zone_positions),))
        modes = np.empty(shape=(len(zone_positions),))
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
        if self.states.is_passing:
            # # ball update with the same velocity
            # self.states.is_passing[STATE_LOOKUP['BALL']
            #            ] = self.states.is_passing[STATE_LOOKUP['BALL']] + self.states.ball_vel
            # # TODO if decision == DESICION_LOOKUP['SHOOT'] or decision == DESICION_LOOKUP['PASS'] return negative reward

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
            for key, value in self.off_def_closest_map.items():
                if value['distance'] <= self.stolen_radius:
                    if length(def2oldball_vecs[value['idx']], axis=0) <= length(off2oldball_vecs[key], axis=0):
                        candidates.append(value['idx'])

            # 3. if any in (2), check is any defender able to fetch ball (depend on defender's wingspan_radius), then go (5)
            def_zone_idx = None
            if len(candidates) != 0:
                def_zone_idx, _ = self._find_zone_idx(
                    next_ball_pos, ball_pos, self.states.ball_vel, self.states.defense_positions[candidates], self.wingspan_radius)
            # 5. if any candidates, choose the best catcher among them
                if def_zone_idx is not None:
                    # assign catcher pos to ball pos
                    self.states.update_ball(self.states.defense_positions[candidates][def_zone_idx],
                                            None, False, [0, 0])
                    # TODO if if_def_catch_ball:
                    # game ends with negative reward to offense, positive reward to defense
                    self.states.game_over(REASON_LOOKUP['CAPTURED'])
            # 4. if none in (2), check is any offender able to fetch ball (depend on offender's wingspan_radius), then go (5)
            off_zone_idx = None
            if len(candidates) == 0 or (def_zone_idx is None):
                off_zone_idx, _ = self._find_zone_idx(
                    next_ball_pos, ball_pos, self.states.ball_vel, self.states.offense_positions, self.wingspan_radius)
            # 5. if any candidates, choose the best catcher among them
                if off_zone_idx is not None:
                    # assign catcher pos to ball pos
                    self.states.update_ball(self.states.offense_positions[off_zone_idx],
                                            off_zone_idx, False, [0, 0])
                    logger.info('[BALL] Successfull Pass :D')
            # 6. if none candidates, ball keep flying
            if def_zone_idx is None and off_zone_idx is None:
                self.states.update_ball(self.states.ball_position + self.states.ball_vel,
                                        None, True, self.states.ball_vel)
                logger.info('[BALL] Flying')
        elif decision == DESICION_LOOKUP['SHOOT'] or decision == DESICION_LOOKUP['NO_OP']:
            # dribble, assign ball handler's position to ball
            self.states.update_ball(self.states.offense_positions[self.states.ball_handler_idx],
                                    self.states.ball_handler_idx, False, [0, 0])
        elif decision == DESICION_LOOKUP['PASS']:
            assert self.states.is_passing == False
            ball_pass_dir = np.clip(
                ball_pass_dir, -np.pi, np.pi)
            new_vel = [self.ball_passing_speed * np.cos(ball_pass_dir),
                       self.ball_passing_speed * np.sin(ball_pass_dir)]
            self.states.update_ball(self.states.ball_position + new_vel,
                                    None, True, new_vel)
            logger.info('[BALL] Flying')

    def _calculate_reward(self):
        pass

    def update_closest_map(self):
        for idx, off_pos in enumerate(self.states.offense_positions):
            distances = distance(self.states.defense_positions, off_pos)
            temp_idx = np.argmin(distances)
            self.off_def_closest_map[idx] = {
                'idx': temp_idx, 'distance': distances[temp_idx]}


class States(object):
    """ modulize all states into class """

    def __init__(self):
        """
        """
        # ball and players [ball, offense, defense]
        self.turn = None
        self.positions = None
        self.vels = None
        self.done = None
        self.reason = None
        self.steps = None
        # self.__dirs = None
        # ball state
        self.is_passing = None
        self.ball_handler_idx = None

    def reset(self, positions, vels, ball_handler_idx):
        self.turn = FLAG_LOOPUP['OFFENSE']
        self.positions = positions
        self.vels = vels
        self.done = False
        self.reason = None
        self.steps = 0
        # self.__dirs = None
        # ball state
        self.ball_handler_idx = ball_handler_idx
        self.is_passing = False

    def update_ball(self, pos, ball_handler_idx, is_passing, velocity):
        self.ball_handler_idx = ball_handler_idx
        self.is_passing = is_passing
        self.positions[STATE_LOOKUP['BALL']] = pos
        self.vels[STATE_LOOKUP['BALL']] = np.array(velocity)

    def update_player(self, team_id, pl_idx, position, vel):
        self.positions[team_id][pl_idx] = position
        self.vels[team_id][pl_idx] = vel

    def take_turn(self):
        self.turn = FLAG_LOOPUP['DEFENSE'] if self.turn == FLAG_LOOPUP['OFFENSE'] else FLAG_LOOPUP['OFFENSE']
        self.steps = self.steps + 1

    def game_over(self, reason):
        self.done = True
        self.reason = reason

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
REASON_LOOKUP = {
    'SHOOT': 0,  # A shoot decision is made
    'CAPTURED': 1,  # A defender gets possession of the ball
    'OOB': 2,  # The ball is Out Of Bounds
    'OOT': 3  # Out Of Time by a max time limitation
}


MODE_LOOKUP = {
    'LEAVE': 0,
    'COME': 1,
    'IN': 2,
    'COME-THEN-IN': 3,
    'IN-THEN-OUT': 4
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
    # Tuple(Discrete(3), Box(), Box(5, 2))
    'DECISION': 0,
    'BALL': 1,
    'DASH': 2
}

FLAG_LOOPUP = {
    'OFFENSE': 0,
    'DEFENSE': 1
}

DESICION_LOOKUP = {
    # Discrete(3)
    'SHOOT': 0,
    'PASS': 1,
    'NO_OP': 2
}
