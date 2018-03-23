""" contour plot by using plotly
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from bball_strategies import gym_bball
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go


def get_shoot_action():
    return tuple((  # shoot decision
        np.array(0),
        np.array(0),
        np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    ))


def get_init_pos(off_pos, def_pos):
    return np.array([
        np.array(off_pos),
        np.array([
            [0, 0],
            [0, 0],
            off_pos,
            [0, 0],
            [0, 0]
        ], dtype=np.float),
        np.array([
            [0, 25],
            [0, 25],
            def_pos,
            [0, 25],
            [0, 25]
        ], dtype=np.float)
    ])


def env_init_wrapper(env, init_positions):
    env.init_mode = 2  # customized position
    env.init_positions = init_positions
    env.init_ball_handler_idx = 2
    return env


def rewards():
    court_length = 94
    court_width = 50
    env = gym.make('bball-ndef-v0')
    off_pos = [65, 25]
    rewards_table = np.empty(shape=[court_length, court_width])

    for i in range(court_length):
        for j in range(court_width):
            init_positions = get_init_pos(off_pos, [i, j])
            shoot_action = get_shoot_action()
            env = env_init_wrapper(env, init_positions)
            env.reset()
            _, reward, done, _ = env.step(shoot_action)
            rewards_table[i, j] = reward
        print(i)

    data = [
        go.Contour(
            z=np.transpose(rewards_table, [1, 0]),
            # colorscale=[[0, 'rgb(148, 0, 211)'], [0.75, 'rgb(75, 0, 130)'], [0.8, 'rgb(0, 0, 255)'], [
            #     0.85, 'rgb(0, 255, 0)'], [0.9, 'rgb(255, 255, 0)'], [0.95, 'rgb(255, 127, 0)'], [1, 'rgb(255, 0 , 0)']],
            colorscale=[[0, 'rgb(148, 0, 211)'], [0.05, 'rgb(75, 0, 130)'], [0.1, 'rgb(0, 0, 255)'], [
                0.15, 'rgb(0, 255, 0)'], [0.2, 'rgb(255, 255, 0)'], [0.25, 'rgb(255, 127, 0)'], [1, 'rgb(255, 0 , 0)']],
            contours=dict(
                coloring='heatmap'
            )
        )
    ]
    layout = go.Layout(
        autosize=False,
        width=950,
        height=500
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='reward_contour_plot.html')


def rewards_move_off():
    court_length = 94
    court_width = 50
    env = gym.make('bball-ndef-v0')
    def_pos = [75, 25]
    rewards_table = np.empty(shape=[court_length, court_width])

    for i in range(court_length):
        for j in range(court_width):
            init_positions = get_init_pos([i, j], def_pos)
            shoot_action = get_shoot_action()
            env = env_init_wrapper(env, init_positions)
            env.reset()
            _, reward, done, _ = env.step(shoot_action)
            rewards_table[i, j] = reward
        print(i)

    data = [
        go.Contour(
            z=np.transpose(rewards_table, [1, 0]),
            # colorscale=[[0, 'rgb(148, 0, 211)'], [0.75, 'rgb(75, 0, 130)'], [0.8, 'rgb(0, 0, 255)'], [
            #     0.85, 'rgb(0, 255, 0)'], [0.9, 'rgb(255, 255, 0)'], [0.95, 'rgb(255, 127, 0)'], [1, 'rgb(255, 0 , 0)']],
            colorscale=[[0, 'rgb(148, 0, 211)'], [0.05, 'rgb(75, 0, 130)'], [0.1, 'rgb(0, 0, 255)'], [
                0.15, 'rgb(0, 255, 0)'], [0.2, 'rgb(255, 255, 0)'], [0.25, 'rgb(255, 127, 0)'], [1, 'rgb(255, 0 , 0)']],
            contours=dict(
                coloring='heatmap'
            )
        )
    ]
    layout = go.Layout(
        autosize=False,
        width=950,
        height=500
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='reward_contour_plot.html')

if __name__ == '__main__':
    # rewards()
    rewards_move_off()
