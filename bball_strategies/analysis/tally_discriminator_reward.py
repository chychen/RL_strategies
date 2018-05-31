from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go


def vis_line_chart(real_rewards, fake_rewards, root_path, post_fix):
    """
    real_rewards : float, shape=(num_of_episode, episode_lenth)
        rewards is formatted state by state
    fake_rewards : float, shape=(num_of_episode, episode_lenth)
        rewards is formatted state by state
    root_path : str, 
        line charts will be saved into the '{root_path}/line_chart'
        (create new folder if not exist)
    post_fix: str
        post_fix to the line chart file
    """

    save_path = os.path.join(
        root_path, 'line_chart/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_trace = []
    all_trace.append(
        go.Scatter(
            x=np.arange(0, real_rewards.shape[1]),
            y=np.mean(real_rewards, axis=0),
            name='real defense'
        )
    )
    all_trace.append(
        go.Scatter(
            x=np.arange(0, fake_rewards.shape[1]),
            y=np.mean(fake_rewards, axis=0),
            name='fake defense'
        )
    )

    layout = go.Layout(
        title='mean_reward_from_discriminator',
        xaxis=dict(title='Frame index'),
        yaxis=dict(title='Reward')
    )
    fig = go.Figure(data=all_trace, layout=layout)
    py.plot(fig, filename=os.path.join(
        save_path, 'gail_reward_{}.html'.format(post_fix)), auto_open=False)


def main():

    vis_line_chart(np.ones([100, 50]), np.zeros(
        [100, 50]), os.getcwd(), 'yeah')


if __name__ == '__main__':
    main()
