from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


FPS = 5
OBSERVATION_LENGTH = 10

NON_OVERLAP_LENGTH = 10
ENV_CONDITION_LENGTH = 50


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
    np.save('GAILEnvData.npy', gail_env_data)
    print('GAILEnvData', gail_env_data.shape)

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
    np.save('GAILTransitionData.npy', gail_tran_data)
    print('GAILTransitionData', gail_tran_data.shape)


if __name__ == '__main__':
    main()
