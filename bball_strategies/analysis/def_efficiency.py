
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def amin_length(fr_def, fr_ball):
    vec = fr_def - fr_ball
    return np.amin(np.sqrt(vec**2))


def statistic_len(ball, defense, length=None):
    min_len_table = np.empty(shape=ball.shape[0:2])
    min_len_table += np.inf
    count_table = np.zeros(shape=[5, ], dtype=np.int32)
    for ep_id, (ep_ball, ep_def) in enumerate(zip(ball, defense)):
        for fr_id, (fr_ball, fr_def) in enumerate(zip(ep_ball, ep_def)):
            if length is not None:
                if fr_id >= length[ep_id]:
                    break
            len_ = amin_length(fr_def, fr_ball)
            min_len_table[ep_id, fr_id] = len_
            for i in range(5):
                count_table[i] += 1 if len_ <= (i+1) else 0
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
    offense = np.reshape(data[:, :, 3:13], [
                         data.shape[0], data.shape[1], 5, 2])
    defense = np.reshape(data[:, :, 13:23], [
                         data.shape[0], data.shape[1], 5, 2])
    statistic_len(ball, defense, target_length)


def fake_wo_shotpenalty():
    data = np.load('../data/results_A_fake_B_wo.npy').reshape([10000, 600, 23])
    target_length = np.load('../data/FULL-LEN.npy')
    target_length = np.repeat(target_length, 100)
    ball = np.reshape(data[:, :, 0:2], [data.shape[0], data.shape[1], 1, 2])
    offense = np.reshape(data[:, :, 3:13], [
                         data.shape[0], data.shape[1], 5, 2])
    defense = np.reshape(data[:, :, 13:23], [
                         data.shape[0], data.shape[1], 5, 2])
    statistic_len(ball, defense, target_length)


def main():
    data = np.load('../data/FEATURES-4.npy')
    print(data.shape)
    ball = data[:, :, 0:1, 0:2]
    offense = data[:, :, 1:6, 0:2]
    defense = data[:, :, 6:11, 0:2]
    statistic_len(ball, defense)
    


if __name__ == '__main__':
    main()
    # fake_wi_shotpenalty()
    # fake_wo_shotpenalty()
