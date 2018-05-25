from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


FPS = 5
OBSERVATION_LENGTH = 10

NON_OVERLAP_LENGTH = 10
ENV_CONDITION_LENGTH = 11


def main():
    # load file
    data = np.load('FixedFPS5.npy')
    data_len = np.load('FixedFPS5Length.npy')
    print(data.shape)  # shape=[num_episode, max_length, 11, 4]
    print(data_len.shape)

    # GAIL Env Data
    gail_env_data = []
    for i in range(data.shape[0]):
        for len_idx in range(0, data_len[i]-ENV_CONDITION_LENGTH, NON_OVERLAP_LENGTH):
            temp = data[i, len_idx: len_idx+ENV_CONDITION_LENGTH, :, 0:2]
            gail_env_data.append(temp)
    gail_env_data = np.array(gail_env_data)
    np.save('GAILEnvData_'+str(ENV_CONDITION_LENGTH)+'.npy', gail_env_data)
    print('GAILEnvData', gail_env_data.shape)
    print('Saved')

    # GAIL Transition Data
    # padding by duplicating first frame
    pad = np.concatenate([data[:, 0:1]
                          for _ in range(OBSERVATION_LENGTH-1)], axis=1)
    pad_data = np.concatenate([pad, data], axis=1)
    pad_data_len = data_len + OBSERVATION_LENGTH-1
    gail_tran_data = []
    for i in range(pad_data.shape[0]):
        for chunk_idx in range(0, pad_data_len[i]-ENV_CONDITION_LENGTH-OBSERVATION_LENGTH, NON_OVERLAP_LENGTH):
            chunk = []
            for k in range(chunk_idx, chunk_idx+ENV_CONDITION_LENGTH, 1):
                buffer = pad_data[i, k:k+OBSERVATION_LENGTH, :, 0:2]
                chunk.append(buffer)
            gail_tran_data.append(chunk)
    gail_tran_data = np.array(gail_tran_data)
    print('gail_tran_data', gail_tran_data.shape)  # (25796, 50, 10, 11, 2)

    gail_def_vel = []
    for i in range(gail_tran_data.shape[0]):
        temp = []
        for j in range(gail_tran_data.shape[1]-1):
            def_velocity = gail_tran_data[i, j+1, -1, 6:,
                                          :] - gail_tran_data[i, j, -1, 6:, :]
            temp.append(def_velocity)
        temp.append(def_velocity)  # padding
        gail_def_vel.append(temp)
    gail_def_vel = np.array(gail_def_vel)

    gail_def_action = []
    for i in range(gail_def_vel.shape[0]):
        temp = []
        for j in range(gail_def_vel.shape[1]-1):
            def_acc = gail_def_vel[i, j+1] - gail_def_vel[i, j]
            temp.append(def_acc)
        temp.append(def_acc)  # padding
        gail_def_action.append(temp)
    gail_def_action = np.array(gail_def_action)
    print('gail_def_action', gail_def_action.shape)  # (25796, 50, 10, 11, 2)

    # store
    dict_ = {}
    dict_['OBS'] = gail_tran_data
    dict_['DEF_ACT'] = gail_def_action
    np.save('GAILTransitionData_'+str(ENV_CONDITION_LENGTH)+'.npy', dict_)
    print('Saved')


if __name__ == '__main__':
    main()
