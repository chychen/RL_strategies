""" To analyze the physics limitaion of NBA players and ball
######## SPEED ########
--BALL--
        ### feet ###
                mean: 9.8898809414
                stddev: 9.68270031134
                maximum: 274.923652593
                minimum: 0.0
                mean + 3 * stddev: 38.9379818754
                mean + 3 * stddev per frame: 7.78759637508
        ### meter ###
                mean: 3.01443571094
                stddev: 2.9512870549
                maximum: 83.7967293102
                minimum: 0.0
                mean + 3 * stddev: 11.8682968756
                mean + 3 * stddev per frame: 2.37365937513
--OFFENSIVE PLAYERS--
        ### feet ###
                mean: 4.8631853515
                stddev: 6.31134758487
                maximum: 266.70375004
                minimum: 0.0
                mean + 3 * stddev: 23.7972281061
                mean + 3 * stddev per frame: 4.75944562122
        ### meter ###
                mean: 1.48229889514
                stddev: 1.92369874387
                maximum: 81.2913030121
                minimum: 0.0
                mean + 3 * stddev: 7.25339512674
                mean + 3 * stddev per frame: 1.45067902535
--DEFENSIVE PLAYERS--
        ### feet ###
                mean: 4.03610800751
                stddev: 4.56436216669
                maximum: 236.550839273
                minimum: 0.00192093727118
                mean + 3 * stddev: 17.7291945076
                mean + 3 * stddev per frame: 3.54583890152
        ### meter ###
                mean: 1.23020572069
                stddev: 1.39121758841
                maximum: 72.1006958105
                minimum: 0.000585501680257
                mean + 3 * stddev: 5.40385848591
                mean + 3 * stddev per frame: 1.08077169718
##### ACCERLATION #####
--BALL--
        ### feet ###
                mean: 7.21773150016
                stddev: 9.34066889272
                maximum: 468.660892453
                minimum: 0.0
                mean + 3 * stddev: 35.2397381783
                mean + 3 * stddev per frame: 7.04794763567
        ### meter ###
                mean: 2.19996456125
                stddev: 2.8470358785
                maximum: 142.84784002
                minimum: 0.0
                mean + 3 * stddev: 10.7410721968
                mean + 3 * stddev per frame: 2.14821443935
--OFFENSIVE PLAYERS--
        ### feet ###
                mean: 1.79004387352
                stddev: 7.59701707495
                maximum: 269.823520445
                minimum: 0.0
                mean + 3 * stddev: 24.5810950984
                mean + 3 * stddev per frame: 4.91621901967
        ### meter ###
                mean: 0.545605372649
                stddev: 2.31557080444
                maximum: 82.2422090315
                minimum: 0.0
                mean + 3 * stddev: 7.49231778598
                mean + 3 * stddev per frame: 1.4984635572
--DEFENSIVE PLAYERS--
        ### feet ###
                mean: 1.67604893772
                stddev: 5.27899130646
                maximum: 242.890393168
                minimum: 0.0
                mean + 3 * stddev: 17.5130228571
                mean + 3 * stddev per frame: 3.50260457142
        ### meter ###
                mean: 0.510859716218
                stddev: 1.60903655021
                maximum: 74.0329918377
                minimum: 0.0
                mean + 3 * stddev: 5.33796936685
                mean + 3 * stddev per frame: 1.06759387337
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

FEET_TO_METER = 0.3048
METER_TP_FEET = 1.0 / 0.3048


def analize_passing_speed(data):
    # ball_passing_speed = []
    # ball_positions = np.stack(
    #     [data[:, :, 0:1, 0], data[:, :, 0:1, 1]], axis=-1)
    # players_positions = np.stack(
    #     [data[:, :, 1:, 0], data[:, :, 1:, 1]], axis=-1)
    # print(ball_positions.shape)
    # print(players_positions.shape)
    # pl2ball_vec = players_positions - ball_positions
    # print(pl2ball_vec.shape)
    # pl2ball_length = np.sqrt(
    #     pl2ball_vec[:, :, :, 0]**2 + pl2ball_vec[:, :, :, 1]**2)
    # print(pl2ball_length.shape)
    # k = True
    # for i in range(10):
    #     k = np.logical_and(k, pl2ball_length[:, :, i] > 0)
    # print(np.array(k).shape)
    # indices = np.argwhere(k)
    # indicesindices = np.argwhere((indices[1:, 1] - indices[:-1, 1]) == 1)
    # print(indicesindices.shape)
    # indices = indices[indicesindices[:, 0]]
    # print(indices.shape)
    # ball_positions = ball_positions[indices[:, 0], indices[:, 1]]
    # print(ball_positions.shape)
    # speed_vec = ball_positions[1:] - ball_positions[:-1]
    # print(speed_vec.shape)
    # speed = np.sqrt(speed_vec[:, 0, 0]**2 + speed_vec[:, 0, 1]**2) * FPS
    # print(speed.shape)
    
    # ball
    speed_x = (data[:, 1:, 0, 0] - data[:, :-1, 0, 0]) * FPS
    speed_y = (data[:, 1:, 0, 1] - data[:, :-1, 0, 1]) * FPS
    speed = np.sqrt(speed_x**2 + speed_y**2)

    ball_speed = go.Histogram(
        name='ball_speed',
        x=speed.reshape([-1]),
        opacity=0.75
        # xbins=dict(
        #     start=0.0,
        #     end=50.0,
        #     size=0.5
        # )
    )
    # offense
    speed_x = (data[:, 1:, 1:6, 0] - data[:, :-1, 1:6, 0]) * FPS
    speed_y = (data[:, 1:, 1:6, 1] - data[:, :-1, 1:6, 1]) * FPS
    speed = np.sqrt(speed_x**2 + speed_y**2)

    offense_speed = go.Histogram(
        name='offense_speed',
        x=speed.reshape([-1]),
        opacity=0.75
        # xbins=dict(
        #     start=0.0,
        #     end=50.0,
        #     size=0.5
        # )
    )

    data = [ball_speed, offense_speed]
    layout = go.Layout(barmode='overlay')
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='Histogram.html')


def analyze(x, y):
    v = np.sqrt(x * x + y * y)
    print('\t### feet ###')
    print('\t\tmean:', np.mean(v))
    print('\t\tstddev:', np.std(v))
    print('\t\tmaximum:', np.amax(v))
    print('\t\tminimum:', np.amin(v))
    print('\t\tmean + 3 * stddev:', np.mean(v) + np.std(v) * 3)
    print('\t\tmean + 3 * stddev per frame:',
          (np.mean(v) + np.std(v) * 3) / FPS)
    print('\t### meter ###')
    print('\t\tmean:', np.mean(v) * FEET_TO_METER)
    print('\t\tstddev:', np.std(v) * FEET_TO_METER)
    print('\t\tmaximum:', np.amax(v) * FEET_TO_METER)
    print('\t\tminimum:', np.amin(v) * FEET_TO_METER)
    print('\t\tmean + 3 * stddev:', (np.mean(v) + np.std(v) * 3) * FEET_TO_METER)
    print('\t\tmean + 3 * stddev per frame:',
          ((np.mean(v) + np.std(v) * 3) / FPS) * FEET_TO_METER)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str,
                        help='data name', default='FrameRate5')
    parser.add_argument('--fps', type=float, help='fps', default=5.0)
    args = parser.parse_args()
#     data = np.load('../data/' + args.name + '.npy').reshape([1,-1,11,3])
    data = np.load('../data/' + args.name + '.npy')
    global FPS
    FPS = args.fps
    print(data.shape)
    # (11863, 100, 11, 4)
    speed_x = (data[:, :-1, :, 0] - data[:, 1:, :, 0]) * FPS
    speed_y = (data[:, :-1, :, 1] - data[:, 1:, :, 1]) * FPS
    # speed_z = data[:, :-1, :, 2] - data[:, 1:, :, 2] * FPS

    # print('######## SPEED ########')
    # print('--BALL--')
    # analyze(speed_x[:, :, 0:1], speed_y[:, :, 0:1])
    # print('--OFFENSIVE PLAYERS--')
    # analyze(speed_x[:, :, 1:6], speed_y[:, :, 1:6])
    # print('--DEFENSIVE PLAYERS--')
    # analyze(speed_x[:, :, 6:11], speed_y[:, :, 6:11])

    # acc_x = speed_x[:, :-1] - speed_x[:, 1:]
    # acc_y = speed_y[:, :-1] - speed_y[:, 1:]

    # print('##### ACCERLATION #####')
    # print('--BALL--')
    # analyze(acc_x[:, :, 0:1], acc_y[:, :, 0:1])
    # print('--OFFENSIVE PLAYERS--')
    # analyze(acc_x[:, :, 1:6], acc_y[:, :, 1:6])
    # print('--DEFENSIVE PLAYERS--')
    # analyze(acc_x[:, :, 6:11], acc_y[:, :, 6:11])

    analize_passing_speed(data)


if __name__ == '__main__':
    main()
