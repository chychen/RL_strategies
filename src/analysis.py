from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


def main():
    data = np.load('../data/FEATURES-4.npy')
    print(data.shape)
    # (11863, 100, 11, 4)
    # fps = 6.25
    speed_x = (data[:, :99, :, 0] - data[:, 1:, :, 0]) / 0.16 * 0.3048
    speed_x = speed_x[:, :98] - speed_x[:, 1:]
    speed_y = (data[:, :99, :, 1] - data[:, 1:100, :, 1]) / 0.16 * 0.3048
    speed_y = speed_y[:, :98] - speed_y[:, 1:]
    # speed_z = data[:, :99, :, 2] - data[:, 1:100, :, 2] / 0.16

    # players' average velocity
    velocity = np.sqrt(speed_x * speed_x + speed_y * speed_y)
    print(velocity.shape)
    # 0.3048m = 1ft
    mean_velocity = np.mean(velocity[:, :, 1:])
    stddev_velocity = np.std(velocity[:, :, 1:])
    amax_velocity = np.amax(velocity[:, :, 1:])
    amin_velocity = np.amin(velocity[:, :, 1:])
    print(mean_velocity)
    print(stddev_velocity)
    print(amax_velocity)
    print(amin_velocity)

    # players' velocity
    # ball's velocity
    # players' accelerator
    # ball's accelerator


if __name__ == '__main__':
    main()
