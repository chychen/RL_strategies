# coding=utf-8
""" Translate real sequencial data to state-action dataset for policy supervised pretraining.
    1. 刪測資：判斷罰球(看一開始佔位)，出界發球(看一開始球位置)，episode太短
    2. 截短：球出界，出手後球到籃框，跳frame(影片不連續)
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

"""
data format:
[
    data_num,
    frame_num,
    [ball, offense1, ..., offense5, defense1, ..., defense5],
    [pos_x, pos_y, pos_z, player_pos]
]
"""
ball_idx = 0
offense_idx = range(1, 6)
defense_idx = range(6, 11)
x_idx, y_idx, z_idx, player_pos_idx = range(0, 4)

mse_threshold = 10


def dist_squared(p0, p1):
    return (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2


def vis_frame(frame_data):
    img = plt.imread("../gym_bball/envs/fullcourt.png")
    plt.imshow(img, extent=[0.0, 93.9, 0.0, 50.0])
    plt.scatter(frame_data[ball_idx][x_idx], frame_data[ball_idx][y_idx], c='g', marker='o')
    for player_idx in offense_idx:
        plt.scatter(frame_data[player_idx][x_idx], frame_data[player_idx][y_idx], c='r', marker='o')
    for player_idx in defense_idx:
        plt.scatter(frame_data[player_idx][x_idx], frame_data[player_idx][y_idx], c='b', marker='o')
    plt.show()


def main():
    frame_discontinuous_arr = []
    ball_stolen_arr = []
    ball_outside_arr = []
    ball_shot_arr = []
    air_ball_shot_arr = []
    start_outside_arr = []
    start_free_throw_arr = []
    episode_too_short_arr = []

    # load file
    data = np.load('FPS5.npy')
    data_len = np.load('FPS5Length.npy')
    print(data.shape)
    print(data_len.shape)

    # filter
    for data_idx in range(data.shape[0]):

        # filter out frame length < 25 (5 seconde)
        if data_len[data_idx] <= 25:
            episode_too_short_arr.append(data_idx)
            continue

        for frame_idx in range(data_len[data_idx]):
            ball_x = data[data_idx][frame_idx][ball_idx][x_idx]
            ball_y = data[data_idx][frame_idx][ball_idx][y_idx]
            ball_z = data[data_idx][frame_idx][ball_idx][z_idx]

            if frame_idx == 0:
                # start with free throws
                if dist_squared([94 - 19, 50 / 2], [ball_x, ball_y]) < 6 ** 2:

                    # check player position
                    def check_player_position(players, is_offensive):
                        free_throw_line_player = []
                        three_point_line_player = []
                        rebound_player = []

                        for player_idx in players:
                            player_x = data[data_idx][frame_idx][player_idx][x_idx]
                            player_y = data[data_idx][frame_idx][player_idx][y_idx]

                            if dist_squared([94.0 - 5.25, 50 / 2], [player_x, player_y]) >= 23.75 ** 2 or \
                                    player_y <= 3 or player_y >= 47:
                                three_point_line_player.append(player_idx)

                            elif 94 - 4 - 15 <= player_x <= 94 - 4 and \
                                    (17 - 3 <= player_y <= 17 or 17 + 16 <= player_y <= 17 + 16 + 3):
                                rebound_player.append(player_idx)

                            elif dist_squared([94 - 19, 50 / 2], [player_x, player_y]) < 6 ** 2 and \
                                    player_x <= 94 - 19:
                                free_throw_line_player.append(player_idx)

                        if is_offensive:
                            return len(free_throw_line_player) == 1 and \
                                   len(three_point_line_player) == 2 and \
                                   len(rebound_player) == 2
                        else:
                            return len(free_throw_line_player) == 0 and \
                                   len(three_point_line_player) == 2 and \
                                   len(rebound_player) == 3

                    if check_player_position(offense_idx, is_offensive=True) and \
                            check_player_position(defense_idx, is_offensive=False):
                        start_free_throw_arr.append(data_idx)
                        break

                    if check_player_position(defense_idx, is_offensive=True) and \
                            check_player_position(offense_idx, is_offensive=False):
                        start_free_throw_arr.append(data_idx)
                        break

                # start with ball outside
                if ball_x < 0.0 or ball_y < 0.0 or ball_x > 94.0 or ball_y > 50.0:
                    start_outside_arr.append(data_idx)
                    break

            else:
                # frame discontinuous
                if frame_idx >= 5:
                    mse = mean_squared_error(data[data_idx][frame_idx], data[data_idx][frame_idx - 1])
                    if mse >= mse_threshold:
                        frame_discontinuous_arr.append(data_idx)
                        break

                cut_flag = False

                # ball is outside
                if ball_x < 0.0 or ball_y < 0.0 or ball_x > 94.0 or ball_y > 50.0:
                    cut_flag = True
                    ball_outside_arr.append(data_idx)

                # ball has been shot / ball is on the top of rim
                # elif dist_squared([94.0 - 5.25, 50 / 2], [ball_x, ball_y]) <= 0.75 ** 2 and ball_z > 10.0:
                elif ball_z > 10.0 and 94 - 4 - 1.5 <= ball_x <= 94 - 4 and 25 - 6 <= ball_y <= 25 + 6:
                    cut_flag = True
                    if dist_squared([94.0 - 5.25, 50 / 2], [ball_x, ball_y]) <= 0.75 ** 2:
                        ball_shot_arr.append(data_idx)
                    else:
                        air_ball_shot_arr.append(data_idx)

                # steal ball
                # TODO: not finish yet
                # if not cut_flag:
                #     min_ball_dist = sys.float_info.max
                #     min_ball_dist_player = 0
                #     for player_idx in range(1, 11):
                #         player_x = data[data_idx][frame_idx][player_idx][x_idx]
                #         player_y = data[data_idx][frame_idx][player_idx][y_idx]
                #         ball_dist = dist_squared([ball_x, ball_y], [player_x, player_y])
                #         if ball_dist < 4**2 and ball_dist < min_ball_dist and ball_z <= 10.0:
                #             min_ball_dist = ball_dist
                #             min_ball_dist_player = player_idx
                #     if 6 <= min_ball_dist_player <= 10:
                #         ball_stolen_arr.append(data_idx)
                #         break

                if cut_flag:
                    prev_data_len = data_len[data_idx]
                    data_len[data_idx] = frame_idx
                    for i in range(frame_idx, prev_data_len):
                        data[data_idx][i].fill(0)
                    break

    # print preprocessed data id
    print("Data removed:")
    print("frame_discontinuous: {}", frame_discontinuous_arr)
    print("ball_stolen: {}", ball_stolen_arr)
    print("start_outside: {}", start_outside_arr)
    print("start_free_throw: {}", start_free_throw_arr)
    print("episode_too_short: {}", episode_too_short_arr)
    print("")
    print("Cut data frames:")
    print("ball_outside: {}", ball_outside_arr)
    print("ball_shot: {}", ball_shot_arr)
    print("air_ball: {}", air_ball_shot_arr)

    # save
    indices = range(len(data))
    remove_indices = start_free_throw_arr + start_outside_arr + frame_discontinuous_arr
    remain_indices = [i for j, i in enumerate(indices) if j not in remove_indices]
    data = data[remain_indices]
    data_len = data_len[remain_indices]
    print(data.shape)
    print(data_len.shape)

    np.save('FixedFPS5.npy', data)
    np.save('FixedFPS5Length.npy', data_len)
    print('Remains {} tranisions'.format(np.sum(data_len)))


if __name__ == '__main__':
    main()
