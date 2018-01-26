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
    self.passing_speed = 30.0 / FPS

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
        'video.frames_per_second': FPS
    }

    def __init__(self):
        # config might be setup by Wrapper
        self.if_init_by_default = False
        self.if_vis_trajectory = False
        self.if_vis_visual_aid = False
        # for render()
        self.viewer = None
        self.def_pl_transforms = None
        self.off_pl_transforms = None
        self.ball_transform = None
        # court information
        self.court_length = 94
        self.court_width = 50
        self.left_basket_pos = [0 + 5.25, 25]
        self.right_basket_pos = [94 - 5.25, 25]
        # this is the distance between two players' center position
        self.screen_radius = 2.0 * 2
        self.wingspan_radius = 3.5
        self.stolen_radius = 5.0
        # physics limitations TODO per frame
        self.pl_max_speed = 38.9379818754 / FPS
        # cost at least one second to cross over the opponent
        self.pl_collision_speed = self.screen_radius / FPS
        self.pl_max_power = 24.5810950984 / FPS
        # Env information
        self.state = None  # Tuple(Box(2,), Box(5, 2), Box(5, 2))
        self.last_state = None  # Tuple(Box(2,), Box(5, 2), Box(5, 2))
        self.ball_state = self.BallState()
        self.off_def_closest_map = None  # dict()
        self.done = None  # boolean

        # must define properties
        self.action_space = self._set_action_space()
        self.observation_space = self._set_observation_space()

        # random seed
        self._seed()

    def _step(self, action):
        """ Following gym.Env.step
        """
        # temp_state for last_state update
        temp_state = copy.deepcopy(self.state)
        if self.last_state is not None:
            velocity_state = np.subtract(self.state, self.last_state)
        else:
            velocity_state = np.subtract(self.state, self.state)
        # action decomposition
        flag = np.argmax(action[ACTION_LOOKUP['FLAG']])
        assert flag.shape == ()
        decision = np.argmax(action[ACTION_LOOKUP['DECISION']])
        ball_pass_dir = action[ACTION_LOOKUP['BALL']]
        off_pl_dash = action[ACTION_LOOKUP['OFFENSE']]
        def_pl_dash = action[ACTION_LOOKUP['DEFENSE']]

        if flag == FLAG_LOOPUP['OFFENSE']:
            result = self._step_offense(
                decision, ball_pass_dir, off_pl_dash, velocity_state)
        elif flag == FLAG_LOOPUP['DEFENSE']:
            result = self._step_defense(
                def_pl_dash, velocity_state)
        else:
            raise KeyError(
                'FLAG #{} have not been defined in FLAG_LOOPUP table'.format(flag))

        # update env information
        self.update_closest_map()
        self.last_state = temp_state
        return result

    def _reset(self):
        """ random init positions in the right half court
        1. init offensive team randomlu
        2. add defensive team next to each offensive player in the basket side.
        """
        if self.if_init_by_default:
            off_players_pos = np.array([
                [80, 40],
                [70, 35],
                [60, 25],
                [70, 15],
                [80, 10]
            ], dtype=np.float)
            ball_handler_idx = 2
        else:
            off_players_pos = self.np_random_generator.uniform(
                low=[self.court_length // 2, 0], high=[self.court_length, self.court_width], size=[5, 2])
            ball_handler_idx = np.floor(self.np_random_generator.uniform(
                low=0.0, high=5.0)).astype(np.int)

        def_players_pos = np.array(off_players_pos, copy=True)
        vec = self.right_basket_pos - off_players_pos
        vec_length = length(vec, axis=1)
        # vec_length = np.sqrt(np.sum(vec * vec, axis=1))
        u_vec = vec / np.stack([vec_length, vec_length], axis=1)
        def_players_pos = def_players_pos + u_vec * self.screen_radius * 2
        ball_pos = np.array(off_players_pos[ball_handler_idx, :], copy=True)
        # reinit Env information
        self.def_pl_transforms = []
        self.off_pl_transforms = []
        self.ball_transform = None
        self.done = False

        self.state = np.array([ball_pos, off_players_pos, def_players_pos])
        self.last_state = None
        self.ball_state.reset(handler_idx=ball_handler_idx, is_passing=False)
        self.off_def_closest_map = dict()
        self.update_closest_map()

        return self.state

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
                        logger.info('{}'.format(self.if_vis_visual_aid))
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
                        logger.info('{}'.format(self.if_vis_visual_aid))
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
            for trans, pos in zip(self.def_pl_transforms, self.state[STATE_LOOKUP['DEFENSE']]):
                trans.set_translation(pos[0], pos[1])
            # offensive players
            for trans, pos in zip(self.off_pl_transforms, self.state[STATE_LOOKUP['OFFENSE']]):
                trans.set_translation(pos[0], pos[1])
            # ball
            ball_pos = self.state[STATE_LOOKUP['BALL']]
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
                pos = self.state[STATE_LOOKUP['DEFENSE']][i]
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
                pos = self.state[STATE_LOOKUP['OFFENSE']][i]
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
            ball_pos = self.state[STATE_LOOKUP['BALL']]
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
            spaces.Discrete(2),  # offense or defense
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

    def _step_offense(self, decision, ball_pass_dir, off_pl_dash, velocity_state):
        """
        Returns
        -------
        observation (object) : agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean) : whether the episode has ended, in which case further step() calls will return undefined results
        info (dict) : contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self._update_player_state(
            off_pl_dash, velocity_state, STATE_LOOKUP['OFFENSE'])
        # update ball state
        self._update_ball_state(decision, ball_pass_dir)

        if decision == DESICION_LOOKUP['SHOOT']:
            reward = self._calculate_reward()
            return self.state, 0.0, self.done, dict()
        elif decision == DESICION_LOOKUP['PASS']:

            return self.state, 0.0, self.done, dict()
        elif decision == DESICION_LOOKUP['NO_OP']:

            return self.state, 0.0, self.done, dict()
        else:
            raise KeyError(
                'Decision #{} have not been defined in DESICION_LOOKUP table'.format(decision))

        # if episode ended
        # --A shoot decision is made,
        # --The ball is out of bounds.
        # --A defender gets possession of the ball.
        # --A max time limit for the episode is reached.

    def _step_defense(self, def_pl_dash, velocity_state):
        """
        Returns
        -------
        observation (object) : agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean) : whether the episode has ended, in which case further step() calls will return undefined results
        info (dict) : contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self._update_player_state(
            def_pl_dash, velocity_state, STATE_LOOKUP['DEFENSE'])
        return self.state, 0.0, False, dict()

    def _update_player_state(self, pl_dash, velocity_state, state_idx):
        """ Update the player's movement following the physics limitation predefined
        Inputs
        ------
        pl_dash : float, shape=[5,2]
            the force of each player
        """
        # 0. update the velocity_state by player's power (with max speed limit = self.pl_max_speed)
        # 1. check if move toward opponent
        # 2. check if move into the opponent's screen range (self.screen_radius)
        # 3. if any collisions, calculate the discounted velocity, then update player
        # 4. if none collisions, update with velocity

        # 0. update the velocity_state by player's power (with max speed limit = self.pl_max_speed)
        # update player state
        assert pl_dash.shape == (5, 2)
        # decomposing into power and direction
        pl_power = np.clip(
            pl_dash[:, DASH_LOOKUP['POWER']], 0, self.pl_max_power)
        pl_dir = np.clip(
            pl_dash[:, DASH_LOOKUP['DIRECTION']], -np.pi, np.pi)
        last_pl_velocity = velocity_state[state_idx]
        pl_acc = np.stack([np.cos(pl_dir) * pl_power,
                           np.sin(pl_dir) * pl_power], axis=1)
        pl_collision_vel = np.stack([np.cos(pl_dir) * self.pl_collision_speed,
                                     np.sin(pl_dir) * self.pl_collision_speed], axis=1)
        assert pl_acc.shape == last_pl_velocity.shape
        pl_velocity = np.add(velocity_state[state_idx], pl_acc)
        pl_speed = length(pl_velocity, axis=1)
        # can't not exceed the limits
        indices = np.argwhere(pl_speed > self.pl_max_speed)
        pl_velocity[indices] = pl_velocity[indices] / \
            np.stack([pl_speed[indices], pl_speed[indices]],
                     axis=-1) * self.pl_max_speed

        # 1. check if move toward opponent
        # 2. check if move into the opponent's screen range (self.screen_radius)
        if state_idx == STATE_LOOKUP['DEFENSE']:
            opp_idx = STATE_LOOKUP['OFFENSE']
        elif state_idx == STATE_LOOKUP['OFFENSE']:
            opp_idx = STATE_LOOKUP['DEFENSE']

        opp_positions = self.state[opp_idx]
        for pl_idx, (pl_pos, next_pl_pos) in enumerate(zip(self.state[state_idx], self.state[state_idx] + pl_velocity)):
            zone_idx = self._find_zone_idx(
                next_pl_pos, pl_pos, pl_velocity[pl_idx], opp_positions, self.screen_radius)
            if zone_idx is not None:
                next_pl_pos = pl_pos + pl_collision_vel[pl_idx]
                # a. length between player and opponent
                # b. length between opponent and velocity vector = a dot dir
                # c. legnth from player to circle = sqrt(aa-bb)-sqrt(rr-(aa-bb))
                # d. final velocity = c * direction + sqrt(rr-(aa-bb)) * self.pl_collision_speed
            self.state[state_idx][pl_idx] = next_pl_pos

    def _find_zone_idx(self, next_pos, pos, vel, zone_positions, zone_radius):
        shortest_dists = np.empty(shape=(len(zone_positions),))
        next_pos2zone_vecs = zone_positions - next_pos
        pos2zone_vecs = zone_positions - pos
        for i, (next_vec, vec) in enumerate(zip(next_pos2zone_vecs, pos2zone_vecs)):
            next_dotvalue = np.dot(next_vec, vel)
            dotvalue = np.dot(vec, vel)
            if next_dotvalue <= 0.0 and dotvalue <= 0.0:  # leave
                shortest_dists[i] = sys.float_info.max
            elif next_dotvalue > 0.0 and dotvalue > 0.0:  # come
                temp_dist = length(next_vec, axis=0)
                if temp_dist <= zone_radius:
                    shortest_dists[i] = temp_dist
                else:
                    shortest_dists[i] = sys.float_info.max
            elif next_dotvalue <= 0.0 and dotvalue > 0.0:  # in then out
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
        return zone_idx

    def _update_ball_state(self, decision, ball_pass_dir):
        """
        Inputs
        ------
        decision : int
            offensive team's decision, including SHOOT, NOOP, and PASS.
        """
        if self.ball_state.is_passing:
            # # ball update with the same velocity
            # self.state[STATE_LOOKUP['BALL']
            #            ] = self.state[STATE_LOOKUP['BALL']] + self.ball_state.velocity
            # # TODO if decision == DESICION_LOOKUP['SHOOT'] or decision == DESICION_LOOKUP['PASS'] return negative reward

            # check if ball caught/stolen
            # Prerequisites:
            # 1. if any defenders is close enough to offender (depend on stolen_radius)
            # 2. defender is closer to ball init position than offender
            # 3. if any in (2), check is any defender able to fetch ball (depend on defender's wingspan_radius), then go (5)
            # 4. if none in (2), check is any offender able to fetch ball (depend on offender's wingspan_radius), then go (5)
            # 5. if any candidates, choose the best catcher among them
            # 6. if none candidates, ball keep flying
            ball_pos = self.state[STATE_LOOKUP['BALL']]
            next_ball_pos = self.state[STATE_LOOKUP['BALL']
                                       ] + self.ball_state.velocity
            off2oldball_vecs = self.state[STATE_LOOKUP['OFFENSE']
                                          ] - ball_pos
            def2oldball_vecs = self.state[STATE_LOOKUP['DEFENSE']
                                          ] - ball_pos

            candidates = []
            for key, value in self.off_def_closest_map.items():
                if value['distance'] <= self.stolen_radius:
                    if length(def2oldball_vecs[value['idx']], axis=0) <= length(off2oldball_vecs[key], axis=0):
                        candidates.append(value['idx'])

            def_zone_idx = None
            if len(candidates) != 0:
                def_zone_idx = self._find_zone_idx(
                    next_ball_pos, ball_pos, self.ball_state.velocity, self.state[STATE_LOOKUP['DEFFENSE']][candidates], self.wingspan_radius)
                if def_zone_idx is not None:
                    # assign catcher pos to ball pos
                    self.state[STATE_LOOKUP['BALL']
                               ] = self.state[STATE_LOOKUP['DEFENSE']][def_zone_idx]
                    self.ball_state.update(
                        handler_idx=def_zone_idx, is_passing=False, velocity=[0, 0])
                    # TODO if if_def_catch_ball:
                    # game ends with negative reward to offense, positive reward to defense
                    logger.info('DEFENSE steals the ball QQ')
                    self.done = True
            off_zone_idx = None            
            if len(candidates) == 0 or (def_zone_idx is None):
                off_zone_idx = self._find_zone_idx(
                    next_ball_pos, ball_pos, self.ball_state.velocity, self.state[STATE_LOOKUP['OFFENSE']], self.wingspan_radius)
                if off_zone_idx is not None:
                    # assign catcher pos to ball pos
                    self.state[STATE_LOOKUP['BALL']
                               ] = self.state[STATE_LOOKUP['OFFENSE']][off_zone_idx]
                    self.ball_state.update(
                        handler_idx=off_zone_idx, is_passing=False, velocity=[0, 0])
                    logger.info('Successfull Pass :D')
            if def_zone_idx is None and off_zone_idx is None:
                self.state[STATE_LOOKUP['BALL']
                           ] = self.state[STATE_LOOKUP['BALL']] + self.ball_state.velocity
                logger.info('Ball is Flying')

        elif decision == DESICION_LOOKUP['SHOOT'] or decision == DESICION_LOOKUP['NO_OP']:
            # dribble, assign ball handler's position to ball
            self.state[STATE_LOOKUP['BALL']
                       ] = self.state[STATE_LOOKUP['OFFENSE']][self.ball_state.handler_idx]
            pass
        elif decision == DESICION_LOOKUP['PASS']:
            assert self.ball_state.is_passing == False
            ball_pass_dir = np.clip(
                ball_pass_dir, -np.pi, np.pi)
            new_vel = [self.ball_state.passing_speed * np.cos(ball_pass_dir),
                       self.ball_state.passing_speed * np.sin(ball_pass_dir)]
            self.ball_state.update(
                handler_idx=None, is_passing=True, velocity=new_vel)
            self.state[STATE_LOOKUP['BALL']
                       ] = self.state[STATE_LOOKUP['BALL']] + self.ball_state.velocity
            logger.info('Ball is Flying')
            
    def _calculate_reward(self):
        pass

    def update_closest_map(self):
        for idx, off_pos in enumerate(self.state[STATE_LOOKUP['OFFENSE']]):
            distances = distance(self.state[STATE_LOOKUP['DEFENSE']], off_pos)
            temp_idx = np.argmin(distances)
            self.off_def_closest_map[idx] = {
                'idx': temp_idx, 'distance': distances[temp_idx]}

    class BallState(object):
        def __init__(self):
            self.is_passing = None
            self.handler_idx = None
            self.velocity = None
            self.passing_speed = 30.0 / FPS

        def reset(self, handler_idx, is_passing=False):
            self.is_passing = is_passing
            self.handler_idx = handler_idx
            self.velocity = np.array([0, 0])

        def update(self, handler_idx, is_passing, velocity):
            self.is_passing = is_passing
            self.handler_idx = handler_idx
            self.velocity = np.array(velocity)


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
    # Tuple(Discrete(2), Discrete(3), Box(), Box(5, 2), Box(5, 2))
    'FLAG': 0,
    'DECISION': 1,
    'BALL': 2,
    'OFFENSE': 3,
    'DEFENSE': 4
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
