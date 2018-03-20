""" 
use vector (x,y) rather than (power, direction)
formulate the defense with two mode, man2man defense and zone defense
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FPS = 5.0


def length(vec, axis=1):
    return np.sqrt(np.sum(vec * vec, axis=axis))


def distance(pos_a, pos_b, axis=1):
    vec = pos_b - pos_a
    return length(vec, axis=axis)


class BBallNDefEnv(gym.Env):
    """ the simulated environment designed for studying the strategies of basketball games.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS * 2.0
    }

    def __init__(self):
        # config might be setup by Wrapper
        self.init_mode = None
        self.init_positions = None  # if_init_by_input
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
        self.time_limit = 24 * FPS
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

        Inputs
        ------
        action : list
            as [decision(1,), ball_dir(2,), offense_dash(5,2), defense_dash(5,2)]

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
        self._update_player_state(
            off_pl_dash, self.states.vels, STATE_LOOKUP['OFFENSE'])
        # update ball state
        self._update_ball_state(decision, ball_pass_dir)
        if decision == DESICION_LOOKUP['SHOOT']:
            self.states.update_status(
                done=True, status=STATUS_LOOKUP['SHOOT'])
        # OOB
        oob_padding = 3
        if self.states.ball_position[0] >= self.court_length + oob_padding or self.states.ball_position[0] < self.court_length / 2 - oob_padding or self.states.ball_position[1] >= self.court_width + oob_padding or self.states.ball_position[1] < 0.0 - oob_padding:
            self.states.update_status(
                done=True, status=STATUS_LOOKUP['OOB'])
        reward = 2.0

        # OOT
        if self.states.steps + 1 >= self.time_limit:  # zero-based
            self.states.update_status(done=True, status=STATUS_LOOKUP['OOT'])

        # termination conditions
        if self.states.done:
            if self.states.status == STATUS_LOOKUP['SHOOT']:
                logger.debug('[GAME OVER], A shoot decision is made')
                if self.states.ball_handler_idx is None:
                    logger.debug('[GAME OVER], No ball handler ...')
                    reward = 1.0
                else:
                    reward += self._calculate_reward()
            elif self.states.status == STATUS_LOOKUP['CAPTURED']:
                logger.debug(
                    '[GAME OVER], A defender gets possession of the ball')
                reward = 0.0
            elif self.states.status == STATUS_LOOKUP['OOB']:
                logger.debug('[GAME OVER], The ball is out of bounds.')
                reward = 1.0
            elif self.states.status == STATUS_LOOKUP['OOT']:
                logger.debug(
                    '[GAME OVER], Max time limit for the episode is reached')
                reward = 0.0
        else:
            # update formulated defense
            def_pl_dash = self._formulated_defense_dash(mode=0)
            self._update_player_state(
                def_pl_dash, self.states.vels, STATE_LOOKUP['DEFENSE'])

            if self.states.status == STATUS_LOOKUP['CATCH']:
                logger.debug('[GAME STATUS] Successfully Pass :D')
                reward += 10.0

        # update env information
        self.states.end_step()

        return self._get_obs(), reward, self.states.done, None

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
        """ Resets the state of the environment and returns an initial observation.

        Returns : self._get_obs()
        """
        if self.init_mode == INIT_LOOKUP['INPUT']:
            assert self.init_positions is not None
            ball_pos = self.init_positions[0]
            off_positions = self.init_positions[1]
            def_positions = self.init_positions[2]
            off2ball_vec = off_positions - ball_pos
            ball_handler_idx = np.argmin(length(off2ball_vec, axis=1))
            self.init_positions[0] = off_positions[ball_handler_idx]
            self.states.reset(
                self.init_positions, ball_handler_idx, buffer_size=self.buffer_size)
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
            u_vec = vec / np.stack([vec_length, vec_length], axis=1)
            def_positions = def_positions + u_vec * self.screen_radius * 2
            ball_pos = np.array(
                off_positions[ball_handler_idx, :], copy=True, dtype=np.float32)
            positions = np.array([ball_pos, off_positions, def_positions])
            self.states.reset(positions, ball_handler_idx,
                              buffer_size=self.buffer_size)

        return self._get_obs()

    def render(self, mode='human'):
        """ Renders the environment.

        if_vis_trajectory : whether to vis the trajectory
        if_vis_visual_aid : whether vis the collision range, fetchable range, and stealable range.
        """
        from gym.envs.classic_control import rendering
        if not self.if_vis_trajectory:
            if self.viewer is None:
                self.viewer = rendering.Viewer(940+100, 500+100)
                # feet # coordinates
                self.viewer.set_bounds(0-5, 94+5, 0-5, 50+5)
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
                self.viewer = rendering.Viewer(940+100, 500+100)
                # feet # coordinates
                self.viewer.set_bounds(0-5, 94+5, 0-5, 50+5)
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
        ActTuple(Discrete(3), Box(2), Box(5, 2))
        """
        return tools.NDefActTuple((
            spaces.Discrete(3),  # offensive decision
            # ball dir
            spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            ),
            # offense player DASH(x, y)
            spaces.Box(
                low=-self.pl_max_power,
                high=self.pl_max_power,
                shape=(5, 2),
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
        low_ = np.array([self.states.x_low_bound, self.states.y_low_bound]) * \
            np.ones(shape=[self.buffer_size, 14, 2])
        high_ = np.array([self.states.x_high_bound, self.states.y_high_bound]) * \
            np.ones(shape=[self.buffer_size, 14, 2])
        return spaces.Box(low=low_, high=high_, dtype=np.float32)

    def _formulated_defense_dash(self, mode=0):
        """ get the formulated defense
        mode 0 : man to man defense
        mode 1 : zone defense
        * for off_player in offense_team_in_order_by_who_has_ball_or_closer_to_the_basket:
            * for havent_matched_def_player in havent_matched_defense_palyers:
                * find the closet havent_matched_def_player to off_payer
                * calculate the acceleration (next def pos is in between the basket and offense you defense

        Return
        ------
        def_pl_dash : float, shape=(5,2)
        """
        def_pl_dash = np.empty([5, 2], dtype=np.float32)
        off_positions = self.states.positions[STATE_LOOKUP['OFFENSE']]

        def update_best_defender():
            def_positions = copy.deepcopy(
                self.states.positions[STATE_LOOKUP['DEFENSE']])
            off_order = []
            if self.states.ball_handler_idx is not None:
                off_order.append(self.states.ball_handler_idx)
            off2basket_lens = length(
                self.right_basket_pos - off_positions, axis=1)
            len_order = np.argsort(off2basket_lens)
            for i in len_order:
                if i not in off_order:
                    off_order.append(i)
            for i, off_pos in enumerate(off_positions[off_order]):
                def2off_len = length(def_positions-off_pos, axis=1)
                best_defenser_idx = np.argmin(def2off_len)
                self.states.off_pair_def[off_order[i]] = best_defenser_idx
                def_positions[best_defenser_idx] = np.inf

        def update_pl_dash():
            def_positions = self.states.positions[STATE_LOOKUP['DEFENSE']]
            for i, off_pos in enumerate(off_positions):
                best_defenser_idx = self.states.off_pair_def[i]
                # find best defensive pos for offense player
                off2basket = self.right_basket_pos - off_pos
                off2basket_len = length(off2basket, axis=0)
                next_best_def_pos = off_pos + \
                    off2basket * (self.screen_radius + 0.5) / off2basket_len
                def_vel = self.states.vels[STATE_LOOKUP['DEFENSE']
                                           ][best_defenser_idx]
                def_pl_dash[best_defenser_idx] = next_best_def_pos - \
                    def_vel - def_positions[best_defenser_idx]

        if mode == 0:  # man to man defense
            if self.states.steps == 0:
                update_best_defender()
            update_pl_dash()
        elif mode == 1:  # zone defense
            update_best_defender()
            update_pl_dash()
        return def_pl_dash

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
        # can't not exceed the acc limits
        pl_dash_len = length(pl_dash, axis=1)
        indices = np.argwhere(pl_dash_len >= self.pl_max_power)
        pl_dash[indices] = pl_dash[indices] / \
            np.stack([pl_dash_len[indices], pl_dash_len[indices]],
                     axis=-1) * self.pl_max_power
        # can't not exceed the speed limits
        pl_vels = np.add(vels[state_idx], pl_dash)
        pl_speed = length(pl_vels, axis=1)
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
                next_pl_pos, pl_pos, pl_vel, opp_positions, self.screen_radius)
        # 3. if any collisions, choose the closest one, calculate the discounted velocity, then update player
            if zone_idx is not None and mode != MODE_LOOKUP['PASS-OUT']:
                pl_collision_vel = [0, 0]  # TODO
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

    def _find_zone_idx(self, next_pos, pos, vel, zone_positions, zone_radius, if_off_ball=False):
        """ find the zone index most reasonable to the vel
        for applications: 
            * who is the most reasonable one to catch ball 
            * who is going to collide to other.

        Inputs
        ------
        next_pos : float, shape=[2,]
            next position of the moving object 
        pos : float, shape=[2,]
            position of the moving object
        vel : float, shape=[2,]
            velocity of the moving object, pos + vel = next_pos
        zone_positions : float, shape=(5, 2)
            the positions of all candidates
        zone_radius : float
            the effective range to find zone index
        if_off_ball : bool
            flag only for offense team while passing ball, which the ball passer could not catch the ball.

        Return
        ------
        zone_idx : int, could be None
            the most reasonable index
        modes : 
            for debug prupose only, what status for each candidates? 
        """
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
            # PASS OUT
            if if_off_ball and i == self.states.last_ball_handler_idx:
                modes[i] = MODE_LOOKUP['PASS-OUT']
                shortest_dists[i] = sys.float_info.max
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
                    logger.debug(
                        '[BALL] You cannot do PASS decision while ball is passing')
                else:
                    ball_pass_dir_len = length(ball_pass_dir, axis=0)
                    new_vel = [self.ball_passing_speed * ball_pass_dir[0] / ball_pass_dir_len,
                               self.ball_passing_speed * ball_pass_dir[1] / ball_pass_dir_len]
                    self.states.update_ball_vel(new_vel)
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
                    self.states.update_status(
                        done=True, status=STATUS_LOOKUP['CAPTURED'])
                    return
            # 4. if none in (2), check is any offender able to fetch ball (depend on offender's wingspan_radius), then go (5)
            off_zone_idx = None
            if len(candidates) == 0 or (def_zone_idx is None):
                off_zone_idx, _ = self._find_zone_idx(
                    next_ball_pos, ball_pos, self.states.ball_vel, self.states.offense_positions, self.wingspan_radius, if_off_ball=True)
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
        # ball and players [ball, offense, defense]
        self.positions = None
        self.buffer_positions = None
        self.vels = None
        self.done = None
        self.status = None
        self.steps = None
        # self.__dirs = None
        # ball state
        self.is_passing = None
        self.last_ball_handler_idx = None
        self.ball_handler_idx = None
        self.off_def_closest_map = None
        self.off_pair_def = None
        # court information
        self.clip_padding = 5.0
        self.x_high_bound = 94.0 + self.clip_padding
        self.x_low_bound = 94.0/2.0 - self.clip_padding
        self.y_high_bound = 50.0 + self.clip_padding
        self.y_low_bound = 0 - self.clip_padding

    def reset(self, positions, ball_handler_idx, buffer_size=5):
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
        self.last_ball_handler_idx = ball_handler_idx
        self.ball_handler_idx = ball_handler_idx
        self.is_passing = False
        self.off_def_closest_map = dict()
        self.update_closest_map()
        self.off_pair_def = np.empty([5, ], dtype=np.int32)

    def update_ball_vel(self, velocity):
        assert self.ball_handler_idx is not None
        self.last_ball_handler_idx = self.ball_handler_idx
        self.ball_handler_idx = None
        self.vels[STATE_LOOKUP['BALL']] = np.array(velocity, dtype=np.float32)

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

    def end_step(self):
        """
        self.buffer_positions : 
        t : index of frame
        """
        for key in STATE_LOOKUP:
            self.buffer_positions[:-1][STATE_LOOKUP[key]
                                       ] = copy.deepcopy(self.buffer_positions[1:][STATE_LOOKUP[key]])
            self.buffer_positions[-1][STATE_LOOKUP[key]
                                      ] = copy.deepcopy(self.positions[STATE_LOOKUP[key]])
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
    'PASS-OUT': 5
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
