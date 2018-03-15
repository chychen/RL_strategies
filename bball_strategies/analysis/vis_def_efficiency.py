""" calculate the mean and stddev of the distance from each offensive player to the closest simulated defensive player, 
    and plot the distance frame by frame for each player compare to the real defensive trajectories.

### three point line
- distance to basket: 23.75 ft
- x: [80,94) y: [3,47)
### paint/ restricted acrea
- x: [75,94) y: [17,33)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

RIGHT_BASKET = [94 - 5.25, 25]
WINGSPAN_RADIUS = 3.5 + 0.5  # to judge who handle the ball
LEN_3PT_BASKET = 23.75 + 5


def get_length(a, b, axis=-1):
    vec = a-b
    return np.sqrt(np.sum(vec*vec, axis=axis))


def evalute_defense(data, length, vis_aid=False):
    """ evaluate the distance to the closest defender for each offensive player on each frames
    and mark the offensive player who has the ball according to WINGSPAN_RADIUS

    Args
    ----
    data : float, shape=[100, 600, 23]
        the positions includes ball, offense, and defense
    length : float, shape=[100,]
        length for each episode

    Returns
    -------
    dist : float, shape=[100, 600, 5]
        the distance to the closest defender for each offensive player
    handler_idx : object, shape=[100, 600, 5]
        one hot vector represent the ball handler idx for each frame
    if_inside_3pt : 
    """
    # shape=[100,600,1,2]
    ball = np.reshape(data[:, :, 0:2], [
        data.shape[0], data.shape[1], 1, 2])
    # shape=[100,600,5,2]
    offense = np.reshape(data[:, :, 3:13], [
        data.shape[0], data.shape[1], 5, 2])
    # shape=[100,600,5,2]
    defense = np.reshape(data[:, :, 13:23], [
        data.shape[0], data.shape[1], 5, 2])

    dist = np.zeros(shape=[data.shape[0], data.shape[1], 5])
    if vis_aid:
        handler_idx = np.empty(
            shape=[data.shape[0], data.shape[1], 5], dtype=object)
        if_inside_3pt = np.empty(
            shape=[data.shape[0], data.shape[1], 5], dtype=object)
        if_inside_paint = np.empty(
            shape=[data.shape[0], data.shape[1], 5], dtype=object)

    for off_idx in range(5):
        offender = offense[:, :, off_idx:off_idx+1, :]
        if vis_aid:
            # mark frames when driible
            indices = np.where(get_length(
                offender[:, :, 0], ball[:, :, 0]) < WINGSPAN_RADIUS)
            handler_idx[indices[0], indices[1], off_idx] = -1
            # check position whether inside the 3pt line
            indices = np.where(get_length(
                offender[:, :, 0], RIGHT_BASKET) < LEN_3PT_BASKET)
            if_inside_3pt[indices[0], indices[1], off_idx] = -2
            # check position whether inside the paint area
            judge_paint = np.logical_and(
                offender[:, :, 0, 0] < 94, offender[:, :, 0, 0] >= 75)
            judge_paint = np.logical_and(
                judge_paint, offender[:, :, 0, 1] < 33)
            judge_paint = np.logical_and(
                judge_paint, offender[:, :, 0, 1] >= 17)
            indices = np.where(judge_paint)
            if_inside_paint[indices[0], indices[1], off_idx] = -3

        # the distance to the closest defender
        dist[:, :, off_idx] = np.amin(get_length(offender, defense), axis=-1)

    # clean up unused length
    for i in range(data.shape[0]):
        dist[i, length[i]:] = 0.0
        if vis_aid:
            handler_idx[i, length[i]:, :] = None
            if_inside_3pt[i, length[i]:, :] = None
            if_inside_paint[i, length[i]:, :] = None

    if vis_aid:
        return dist, handler_idx, if_inside_3pt, if_inside_paint
    else:
        return dist


def plot_by_frames(handler_idx, if_inside_3pt, if_inside_paint, real_dist, fake_wo_dist, fake_wi_dist, length):
    """ plot

    Args
    ----
    handler_idx : int, shape=[100, 600, 5]
        one hot vector represent the ball handler idx for each frame
    if_inside_3pt : 
    if_inside_paint : 
    real_dist : float, shape=[100, 600, 5]
    fake_wo_dist : float, shape=[100, 600, 5]
    fake_wi_dist : float, shape=[100, 600, 5]
    length : float, shape=[100,]
        length for each episode
    """
    if not os.path.exists('plot'):
        os.makedirs('plot')
    for epi_idx in range(handler_idx.shape[0]):
        data = []
        epi_len = length[epi_idx]
        for off_idx in range(5):
            # has ball marker
            trace = go.Scatter(
                x=np.arange(epi_len),
                y=handler_idx[epi_idx, :epi_len, off_idx],
                name='has_ball_'+str(off_idx+1),
                xaxis='x',
                yaxis='y'+str(off_idx+1),
                line=dict(
                    color=('rgb(205, 12, 24)'),
                    width=10)
            )
            data.append(trace)
            # whether the offender inside the 3pt line
            trace = go.Scatter(
                x=np.arange(epi_len),
                y=if_inside_3pt[epi_idx, :epi_len, off_idx],
                name='3pt_'+str(off_idx+1),
                xaxis='x',
                yaxis='y'+str(off_idx+1),
                line=dict(
                    color=('rgb(205, 205, 24)'),
                    width=10)
            )
            data.append(trace)
            # whether the offender inside the paint area
            trace = go.Scatter(
                x=np.arange(epi_len),
                y=if_inside_paint[epi_idx, :epi_len, off_idx],
                name='paint_'+str(off_idx+1),
                xaxis='x',
                yaxis='y'+str(off_idx+1),
                line=dict(
                    color=('rgb(24, 205, 205)'),
                    width=10)
            )
            data.append(trace)
            # real
            trace = go.Scatter(
                x=np.arange(epi_len),
                y=real_dist[epi_idx, :epi_len, off_idx],
                name='real_'+str(off_idx+1),
                xaxis='x',
                yaxis='y'+str(off_idx+1),
                line=dict(
                    color=('rgb(222, 222, 222)'))
            )
            data.append(trace)
            # fake_wo
            trace = go.Scatter(
                x=np.arange(epi_len),
                y=fake_wo_dist[epi_idx, :epi_len, off_idx],
                name='fake_wo_'+str(off_idx+1),
                xaxis='x',
                yaxis='y'+str(off_idx+1),
                line=dict(
                    color=('rgb(12, 205, 24)'))
            )
            data.append(trace)
            # fake_wi
            trace = go.Scatter(
                x=np.arange(epi_len),
                y=fake_wi_dist[epi_idx, :epi_len, off_idx],
                name='fake_wi_'+str(off_idx+1),
                xaxis='x',
                yaxis='y'+str(off_idx+1),
                line=dict(
                    color=('rgb(24, 12, 205)'))
            )
            data.append(trace)
        layout = go.Layout(
            xaxis=dict(title='time (frame)'),
            yaxis1=dict(
                title='player_1\'s distance (feet)',
                domain=[0.0, 0.15]
            ),
            yaxis2=dict(
                title='player_2\'s distance (feet)',
                domain=[0.2, 0.35]
            ),
            yaxis3=dict(
                title='player_3\'s distance (feet)',
                domain=[0.4, 0.55]
            ),
            yaxis4=dict(
                title='player_4\'s distance (feet)',
                domain=[0.6, 0.75]
            ),
            yaxis5=dict(
                title='player_5\'s distance (feet)',
                domain=[0.8, 0.95]
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='plot/epi_{}.html'.format(epi_idx), auto_open=False)


def main():
    real_data = np.load('../data/WGAN/FULL.npy')
    real_data = np.concatenate(
        [
            # ball
            real_data[:, :, 0, :3].reshape(
                [real_data.shape[0], real_data.shape[1], 1 * 3]),
            # team A players
            real_data[:, :, 1:6, :2].reshape(
                [real_data.shape[0], real_data.shape[1], 5 * 2]),
            # team B players
            real_data[:, :, 6:11, :2].reshape(
                [real_data.shape[0], real_data.shape[1], 5 * 2])
        ], axis=-1
    )
    fake_wo_data = np.load(
        '../data/WGAN/results_A_fake_B_wo.npy')[0]
    fake_wi_data = np.load(
        '../data/WGAN/results_A_fake_B.npy')[0]
    target_length = np.load('../data/WGAN/FULL-LEN.npy')
    print(real_data.shape)
    print(fake_wo_data.shape)
    print(fake_wi_data.shape)
    print(target_length.shape)

    # analysis
    real_dist = evalute_defense(real_data, target_length)
    fake_wo_dist = evalute_defense(fake_wo_data, target_length)
    fake_wi_dist, handler_idx, if_inside_3pt, if_inside_paint = evalute_defense(
        fake_wi_data, target_length, vis_aid=True)
    # plot
    plot_by_frames(handler_idx, if_inside_3pt, if_inside_paint, real_dist, fake_wo_dist,
                   fake_wi_dist, target_length)


if __name__ == '__main__':
    main()
