"""
ref: https://www.recunlimited.com/blog/diagrams-basketball-courts/
right hand site court information:
### three point line
- distance to basket: 23.75 ft
- x: [80,94) y: [3,47)
### paint/ restricted acrea
- x: [75,94) y: [17,33)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

MODE_LOOKUP = {
    'NORMAL': 0,
    '3POINT': 1,
    'PAINT': 2
}

RIGHT_BASKET = [94 - 5.25, 25]


def get_length(a, b):
    vec = a - b
    return np.sqrt(np.sum(vec**2))


def amin_length(fr_def, fr_ball):
    vec = fr_def - fr_ball
    return np.amin(np.sqrt(np.sum(vec**2, axis=1)))


def statistic_len(ball, defense, mode=MODE_LOOKUP['NORMAL'], length=None):
    if mode == MODE_LOOKUP['NORMAL']:
        print('MODE: NORMAL')
    elif mode == MODE_LOOKUP['3POINT']:
        print('MODE: 3POINT')
    elif mode == MODE_LOOKUP['PAINT']:
        print('MODE: PAINT')
    min_len_table = np.empty(shape=ball.shape[0:2])
    min_len_table += np.inf
    count_table = np.zeros(shape=[5, ], dtype=np.int32)
    for ep_id, (ep_ball, ep_def) in enumerate(zip(ball, defense)):
        for fr_id, (fr_ball, fr_def) in enumerate(zip(ep_ball, ep_def)):
            if length is not None:
                if fr_id >= length[ep_id]:
                    break

            if mode == MODE_LOOKUP['NORMAL']:
                pass
            elif mode == MODE_LOOKUP['3POINT']:
                ball2baket = get_length(fr_ball, RIGHT_BASKET)
                if ball2baket >= 23.75:
                    continue  # skip
                elif fr_ball[0][0] >= 80 and fr_ball[0][0] < 94 and (fr_ball[0][1] >= 47 or fr_ball[0][1] < 3):
                    continue  # skip
            elif mode == MODE_LOOKUP['PAINT']:
                if fr_ball[0][0] >= 94 or fr_ball[0][0] < 75 or fr_ball[0][1] >= 33 or fr_ball[0][1] < 17:
                    continue  # skip

            len_ = amin_length(fr_def, fr_ball)
            min_len_table[ep_id, fr_id] = len_
            for i in range(5):
                count_table[i] += 1 if len_ <= (i+5) else 0
    indices = np.where(min_len_table != np.inf)
    min_len_mean = np.mean(min_len_table[indices])
    min_len_std = np.std(min_len_table[indices])
    print('min_len_mean: {}'.format(min_len_mean))
    print('min_len_std: {}'.format(min_len_std))
    if length is not None:
        sum_len = np.sum(length)
        for i in range(5):
            print('len < {}, rate(counts/all_frame): {}'.format(i+1,
                                                                count_table[i] / sum_len))
    else:
        for i in range(5):
            print('len < {}, rate(counts/all_frame): {}'.format(i+1,
                                                                count_table[i] / (ball.shape[0] * ball.shape[1])))


def fake_wi_shotpenalty():
    data = np.load('../data/results_A_fake_B.npy').reshape([10000, 600, 23])
    target_length = np.load('../data/FULL-LEN.npy')
    target_length = np.repeat(target_length, 100)
    ball = np.reshape(data[:, :, 0:2], [data.shape[0], data.shape[1], 1, 2])
    # offense = np.reshape(data[:, :, 3:13], [
    #                      data.shape[0], data.shape[1], 5, 2])
    defense = np.reshape(data[:, :, 13:23], [
                         data.shape[0], data.shape[1], 5, 2])
    statistic_len(ball, defense, length=target_length)
    statistic_len(
        ball, defense, mode=MODE_LOOKUP['3POINT'], length=target_length)
    statistic_len(
        ball, defense, mode=MODE_LOOKUP['PAINT'], length=target_length)


def fake_wo_shotpenalty():
    data = np.load('../data/results_A_fake_B_wo.npy').reshape([10000, 600, 23])
    target_length = np.load('../data/FULL-LEN.npy')
    target_length = np.repeat(target_length, 100)
    ball = np.reshape(data[:, :, 0:2], [data.shape[0], data.shape[1], 1, 2])
    # offense = np.reshape(data[:, :, 3:13], [
    #                      data.shape[0], data.shape[1], 5, 2])
    defense = np.reshape(data[:, :, 13:23], [
                         data.shape[0], data.shape[1], 5, 2])
    statistic_len(ball, defense, length=target_length)
    statistic_len(
        ball, defense, mode=MODE_LOOKUP['3POINT'], length=target_length)
    statistic_len(
        ball, defense, mode=MODE_LOOKUP['PAINT'], length=target_length)


def main():
    data = np.load('../data/FEATURES-4.npy')
    print(data.shape)
    ball = data[:, :, 0:1, 0:2]
    # offense = data[:, :, 1:6, 0:2]
    defense = data[:, :, 6:11, 0:2]
    statistic_len(ball, defense)
    statistic_len(
        ball, defense, mode=MODE_LOOKUP['3POINT'])
    statistic_len(
        ball, defense, mode=MODE_LOOKUP['PAINT'])


if __name__ == '__main__':
    print('Real Data')
    main()
    print('Fake WI Open Penalty')
    fake_wi_shotpenalty()
    print('Fake WO Open Penalty')
    fake_wo_shotpenalty()
