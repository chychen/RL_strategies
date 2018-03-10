""" Translate real sequencial data to state-action dataset for policy supervised pretraining.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np


def main():
    data = np.load('FPS5.npy')
    data_len = np.load('FPS5Length.npy')
    print(data.shape)
    print(data_len.shape)
    # filter out frame length < 25 (5 seconde)
    indices = np.argwhere(data_len >= 25)[:, 0]
    print('filter out {} episodes, which length is shorter than 25'.format(data_len.shape[0] - len(indices)))
    data = data[indices]
    data_len = data_len[indices]
    print(data.shape)
    print(data_len.shape)
    # save
    np.save('FixedFPS5.npy', data)
    np.save('FixedFPS5Length.npy', data_len)


if __name__ == '__main__':
    main()
