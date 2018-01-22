""" To analyze the physics limitaion of NBA players and ball
######## SPEED ########
--BALL--
        ### feet ###
                mean: 9.96812766732
                tddev: 9.58744252436
                maximum: 346.746883812
                minimum: 0.0
                mean + 3 * stddev: 38.7304552404
                mean + 3 * stddev per frame: 6.19687283846
        ### meter ###
                mean: 3.038285313
                tddev: 2.92225248142
                maximum: 105.688450186
                minimum: 0.0
                mean + 3 * stddev: 11.8050427573
                mean + 3 * stddev per frame: 1.88880684116
--OFFENSIVE PLAYERS--
        ### feet ###
                mean: 4.82228321969
                tddev: 5.66676430718
                maximum: 356.972509487
                minimum: 0.0
                mean + 3 * stddev: 21.8225761412
                mean + 3 * stddev per frame: 3.4916121826
        ### meter ###
                mean: 1.46983192536
                tddev: 1.72722976083
                maximum: 108.805220892
                minimum: 0.0
                mean + 3 * stddev: 6.65152120785
                mean + 3 * stddev per frame: 1.06424339326
--DEFENSIVE PLAYERS--
        ### feet ###
                mean: 4.0065741936
                tddev: 4.10053463972
                maximum: 295.220443716
                minimum: 0.0
                mean + 3 * stddev: 16.3081781128
                mean + 3 * stddev per frame: 2.60930849804
        ### meter ###
                mean: 1.22120381421
                tddev: 1.24984295819
                maximum: 89.9831912448
                minimum: 0.0
                mean + 3 * stddev: 4.97073268877
                mean + 3 * stddev per frame: 0.795317230204
##### ACCERLATION #####
--BALL--
        ### feet ###
                mean: 6.59227756409
                tddev: 8.59926922199
                maximum: 691.21559203
                minimum: 0.0
                mean + 3 * stddev: 32.3900852301
                mean + 3 * stddev per frame: 5.18241363681
        ### meter ###
                mean: 2.00932620154
                tddev: 2.62105725886
                maximum: 210.682512451
                minimum: 0.0
                mean + 3 * stddev: 9.87249797812
                mean + 3 * stddev per frame: 1.5795996765
--OFFENSIVE PLAYERS--
        ### feet ###
                mean: 1.38937387361
                tddev: 6.35125744014
                maximum: 363.374842672
                minimum: 0.0
                mean + 3 * stddev: 20.443146194
                mean + 3 * stddev per frame: 3.27090339104
        ### meter ###
                mean: 0.423481156676
                tddev: 1.93586326775
                maximum: 110.756652046
                minimum: 0.0
                mean + 3 * stddev: 6.23107095994
                mean + 3 * stddev per frame: 0.99697135359
--DEFENSIVE PLAYERS--
        ### feet ###
                mean: 1.34141704905
                tddev: 4.35347505948
                maximum: 301.58107585
                minimum: 0.0
                mean + 3 * stddev: 14.4018422275
                mean + 3 * stddev per frame: 2.3042947564
        ### meter ###
                mean: 0.408863916551
                tddev: 1.32693919813
                maximum: 91.9219119192
                minimum: 0.0
                mean + 3 * stddev: 4.38968151094
                mean + 3 * stddev per frame: 0.702349041751
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

FPS = 6.25
FEET_TO_METER = 0.3048
METER_TP_FEET = 1.0 / 0.3048


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
#     data = np.load('../data/NBA-ALL.npy')
    data = np.load('../data/FEATURES-4.npy')
    
    print(data.shape)
    # (11863, 100, 11, 4)
    speed_x = (data[:, :-1, :, 0] - data[:, 1:, :, 0]) * FPS
    speed_y = (data[:, :-1, :, 1] - data[:, 1:, :, 1]) * FPS
    # speed_z = data[:, :-1, :, 2] - data[:, 1:, :, 2] * FPS

    print('######## SPEED ########')
    print('--BALL--')
    analyze(speed_x[:, :, 0:1], speed_y[:, :, 0:1])
    print('--OFFENSIVE PLAYERS--')
    analyze(speed_x[:, :, 1:6], speed_y[:, :, 1:6])
    print('--DEFENSIVE PLAYERS--')
    analyze(speed_x[:, :, 6:11], speed_y[:, :, 6:11])

    acc_x = speed_x[:, :-1] - speed_x[:, 1:]
    acc_y = speed_y[:, :-1] - speed_y[:, 1:]

    print('##### ACCERLATION #####')
    print('--BALL--')
    analyze(acc_x[:, :, 0:1], acc_y[:, :, 0:1])
    print('--OFFENSIVE PLAYERS--')
    analyze(acc_x[:, :, 1:6], acc_y[:, :, 1:6])
    print('--DEFENSIVE PLAYERS--')
    analyze(acc_x[:, :, 6:11], acc_y[:, :, 6:11])


if __name__ == '__main__':
    main()
