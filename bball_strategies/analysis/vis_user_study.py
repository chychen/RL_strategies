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
import vis_game
from evaluation import EvaluationMatrix


def vis_user_study(if_vis_game=False):
    """ vis_user_study by line chart analysis and optional plot into vedios
    """
    I_ID = [34, 12, 67, 31, 22, 6, 43, 17, 8]
    C_ID = [333, 878, 453, 265, 1081, 750, 383, 1088, 108]
    N_ID = [23, 66, 74, 47, 92, 43, 9, 92, 5]
    real_data = np.load('../data/WGAN/user_study/results_A_real_B.npy')
    real_data = real_data[C_ID]

    fake_wo_dist = None
    fake_wi_data = np.load(
        '../data/WGAN/user_study/results_A_fake_B.npy')
    fake_wi_data = fake_wi_data[N_ID, C_ID]

    target_length = np.load('../data/WGAN/FULL-LEN.npy')
    target_length = target_length[:len(real_data)]
    target_length[:] = 100

    print(real_data.shape)
    print(fake_wi_data.shape)
    print(target_length.shape)

    evaluator = EvaluationMatrix(
        length=target_length, real_data=real_data, fake_wi_data=fake_wi_data)
    # plot
    evaluator.plot_linechart_distance_by_frames(
        file_name='user_study', mode='THETA')

    # vis game
    if if_vis_game:
        save_path = 'user_study/real'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(len(real_data)):
            vis_game.plot_data(real_data[i], length=100,
                               file_path=save_path+'/play_' + str(i) + '.mp4', if_save=True)
        save_path = 'user_study/fake'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(len(fake_wi_data)):
            vis_game.plot_data(fake_wi_data[i], length=100,
                               file_path=save_path+'/play_' + str(i) + '.mp4', if_save=True)


if __name__ == '__main__':
    vis_user_study(if_vis_game=False)
