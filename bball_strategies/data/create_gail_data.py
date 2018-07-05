from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import h5py
import numpy as np


FPS = 5
OBSERVATION_LENGTH = 1

NON_OVERLAP_LENGTH = 10
ENV_CONDITION_LENGTH_LIST = [12, 22, 32, 42, 52]

RIGHT_BASKET_POS = [94 - 5.25, 25.0]
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0


def clip_state(pos):
    """ make sure no real state out of bound, which is consistent to our environment
    """
    # court information
    clip_padding = 5.0
    x_high_bound = 94.0 + clip_padding
    x_low_bound = 94.0 / 2.0 - clip_padding
    y_high_bound = 50.0 + clip_padding
    y_low_bound = 0 - clip_padding
    pos[:, :, :, 0][np.where(pos[:, :, :, 0] < x_low_bound)] = x_low_bound
    pos[:, :, :, 0][np.where(pos[:, :, :, 0] >= x_high_bound)] = x_high_bound
    pos[:, :, :, 1][np.where(pos[:, :, :, 1] < y_low_bound)] = y_low_bound
    pos[:, :, :, 1][np.where(pos[:, :, :, 1] >= y_high_bound)] = y_high_bound
    return pos


def main():
    # load file
    data = np.load('FixedFPS5.npy')
    data = clip_state(data)
    data_len = np.load('FixedFPS5Length.npy')
    print(data.shape)  # shape=[num_episode, max_length, 11, 4]
    print(data_len.shape)

    for ENV_CONDITION_LENGTH in ENV_CONDITION_LENGTH_LIST:
        # GAIL Env Data
        gail_env_data = []
        for i in range(data.shape[0]):
            for len_idx in range(0, data_len[i] - ENV_CONDITION_LENGTH, NON_OVERLAP_LENGTH):
                temp = data[i, len_idx: len_idx + ENV_CONDITION_LENGTH, :, 0:2]
                temp = np.concatenate([
                    temp,
                    np.tile(RIGHT_BASKET_POS, [ENV_CONDITION_LENGTH, 1, 1]),
                    np.tile([COURT_LENGTH / 2.0, 0.0],
                            [ENV_CONDITION_LENGTH, 1, 1]),
                    np.tile([COURT_LENGTH, COURT_WIDTH],
                            [ENV_CONDITION_LENGTH, 1, 1])
                ], axis=1)
                gail_env_data.append(temp)
        gail_env_data = np.array(gail_env_data)
        np.save('GAILEnvData_' + str(ENV_CONDITION_LENGTH) +
                '.npy', gail_env_data)
        print('GAILEnvData', gail_env_data.shape)
        print('Saved')

        # # GAIL Transition Data
        # # padding by duplicating first frame
        # pad = np.concatenate([data[:, 0:1]
        #                       for _ in range(OBSERVATION_LENGTH-1)], axis=1)
        # pad_data = np.concatenate([pad, data], axis=1)
        # pad_data_len = data_len + OBSERVATION_LENGTH-1
        # gail_tran_data = []
        # for i in range(pad_data.shape[0]):
        #     for chunk_idx in range(0, pad_data_len[i]-ENV_CONDITION_LENGTH-OBSERVATION_LENGTH, NON_OVERLAP_LENGTH):
        #         chunk = []
        #         for k in range(chunk_idx, chunk_idx+ENV_CONDITION_LENGTH, 1):
        #             buffer = pad_data[i, k:k+OBSERVATION_LENGTH, :, 0:2]
        #             chunk.append(buffer)
        #         gail_tran_data.append(chunk)
        # gail_tran_data = np.array(gail_tran_data)
        # print('gail_tran_data', gail_tran_data.shape)  # (25796, 50, 10, 11, 2)

        # GAIL Transition Data
        gail_tran_data = []
        for i in range(data.shape[0]):
            for chunk_idx in range(0, data_len[i] - ENV_CONDITION_LENGTH, NON_OVERLAP_LENGTH):
                # padding by duplicating first frame
                if OBSERVATION_LENGTH > 1:
                    pad = np.concatenate([data[i, chunk_idx:chunk_idx + 1]
                                          for _ in range(OBSERVATION_LENGTH - 1)], axis=0)
                    pad_data = np.concatenate(
                        [pad, data[i, chunk_idx:chunk_idx + ENV_CONDITION_LENGTH]], axis=0)
                else:
                    pad_data = data[i, chunk_idx:chunk_idx +
                                    ENV_CONDITION_LENGTH]
                chunk = []
                for k in range(ENV_CONDITION_LENGTH):
                    buffer = pad_data[k:k + OBSERVATION_LENGTH, :, 0:2]
                    buffer = np.concatenate([
                        buffer,
                        np.tile(RIGHT_BASKET_POS, [OBSERVATION_LENGTH, 1, 1]),
                        np.tile([COURT_LENGTH / 2.0, 0.0],
                                [OBSERVATION_LENGTH, 1, 1]),
                        np.tile([COURT_LENGTH, COURT_WIDTH],
                                [OBSERVATION_LENGTH, 1, 1])
                    ], axis=1)
                    chunk.append(buffer)
                gail_tran_data.append(chunk)
        gail_tran_data = np.array(gail_tran_data)
        # (num_data, length, OBSERVATION_LENGTH, 14, 2)
        print('gail_tran_data', gail_tran_data.shape)

        gail_def_vel = []
        for i in range(gail_tran_data.shape[0]):
            temp = []
            for j in range(gail_tran_data.shape[1] - 1):
                def_velocity = gail_tran_data[i, j + 1, -1, 6:11,
                                              :] - gail_tran_data[i, j, -1, 6:11, :]
                temp.append(def_velocity)
            temp.append(def_velocity)  # padding
            gail_def_vel.append(temp)
        gail_def_vel = np.array(gail_def_vel)

        gail_def_action = []
        for i in range(gail_def_vel.shape[0]):
            temp = []
            for j in range(gail_def_vel.shape[1] - 1):
                def_acc = gail_def_vel[i, j + 1] - gail_def_vel[i, j]
                temp.append(def_acc)
            temp.append(def_acc)  # padding
            gail_def_action.append(temp)
        gail_def_action = np.array(gail_def_action)
        # (num_data, length, 5, 2)
        print('gail_def_action', gail_def_action.shape)

        # store
        init_vel = np.array(gail_tran_data[:, 1, -1, 6:11] - gail_tran_data[:, 0, -1, 6:11])
        print('init_vel', init_vel.shape)
        with h5py.File('GAILTransitionData_' + str(ENV_CONDITION_LENGTH) + '.hdf5', "w") as f:
            dset = f.create_dataset('OBS', data=gail_tran_data[:, 1:]) # first is useless
            dset = f.create_dataset('DEF_ACT', data=gail_def_action[:, :-1])
            dset = f.create_dataset('DEF_INIT_VEL', data=init_vel)
        print('Saved')


if __name__ == '__main__':
    main()
