from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import h5py
import numpy as np


ENV_CONDITION_LENGTH_LIST = ['Testing', 12, 22, 32, 42, 52]


def length(vec, axis):
    return np.sqrt(np.sum(vec**2, axis=axis))


def main():

    for ENV_CONDITION_LENGTH in ENV_CONDITION_LENGTH_LIST:
        data = h5py.File('GAILTransitionData_{}.hdf5'.format(
            ENV_CONDITION_LENGTH), 'r')
        data_obs = data['OBS'].value
        # shape = (?, ENV_CONDITION_LENGTH, 1, 14, 2)
        data_act = data['DEF_ACT'].value
        # shape = (?, ENV_CONDITION_LENGTH, 5, 2)
        data_init_vel = data['DEF_INIT_VEL'].value
        # shape = (?, 5, 2)

        # 1. find the priority of offensive players by summing the distance to ball, then we get an order (off_order).
        off_ball_obs = data_obs[:, :, 0, 0:1]
        off_pl_obs_first = data_obs[:, :, 0, 1:6]
        len_ball_off = np.sum(
            length(off_ball_obs-off_pl_obs_first, axis=-1), axis=1)
        off_order = np.argsort(len_ball_off, axis=-1)
        print(off_order.shape)
        # 2. accumulate distance of the whole episode between 5 offensive players to each defensive player, then we get a 5(def)*5(off) matrix (acc_def_mat).
        off_pl_obs = data_obs[:, :, 0, 1:6]
        def_pl_obs = data_obs[:, :, 0, 6:11]
        acc_def_mat = []
        for def_pl_id in range(5):
            one_def_pl_obs = def_pl_obs[:, :, def_pl_id:def_pl_id+1]
            len_onedef_off = length(off_pl_obs-one_def_pl_obs, axis=-1)
            tmp_sum = np.sum(len_onedef_off, axis=1)
            acc_def_mat.append(tmp_sum)
        acc_def_mat = np.stack(acc_def_mat, axis=1)
        print(acc_def_mat.shape)
        # 3. according to (off_order), we paired offense with the best defender (paired_def_order) from (acc_def_mat) sequencially.
        paired_def_order = []
        for i, off_order_one_epi in enumerate(off_order):
            tmp = []
            for off_id in off_order_one_epi:
                acc_def_one_off = acc_def_mat[i, :, off_id]
                best_def_order = np.argsort(acc_def_one_off, axis=-1)
                for best_def in best_def_order:
                    if best_def in tmp:
                        continue
                    else:
                        tmp.append(best_def)
                        break
            paired_def_order.append(tmp)
        paired_def_order = np.array(paired_def_order)
        print(paired_def_order.shape)
        # 4. make sure data_act and data_init_vel are consistent with data_obs
        data_defense_obs = data_obs[:, :, :, 6:11]
        ori_data_defense_obs = np.array(data_obs[:, :, :, 6:11])
        ori_data_act = np.array(data_act)
        ori_data_init_vel = np.array(data_init_vel)
        for i in range(data_defense_obs.shape[0]):
            for def_id in range(5):
                data_defense_obs[i, :, :, off_order[i, def_id]
                                 ] = ori_data_defense_obs[i, :, :, paired_def_order[i, def_id]]
                data_act[i, :, off_order[i, def_id]] = ori_data_act[i,
                                                                    :, paired_def_order[i, def_id]]
                data_init_vel[i, off_order[i, def_id]
                              ] = ori_data_init_vel[i, paired_def_order[i, def_id]]
        assert (data_obs[:, :, :, 6:11] != ori_data_defense_obs).any(
        ), "data_obs(defense) should be modified"
        assert (data_act != ori_data_act).any(), "data_act should be modified"
        assert (data_init_vel != ori_data_init_vel).any(
        ), "data_init_vel should be modified"
        # 5. finally, we use (off_order) to permute the index in data.
        data_offense_obs = data_obs[:, :, :, 1:6]
        data_defense_obs = data_obs[:, :, :, 6:11]
        ori_data_offense_obs = np.array(data_obs[:, :, :, 1:6])
        ori_data_defense_obs = np.array(data_obs[:, :, :, 6:11])
        ori_data_act = np.array(data_act)
        ori_data_init_vel = np.array(data_init_vel)
        for i in range(data_defense_obs.shape[0]):
            for off_id in range(5):
                data_offense_obs[i, :, :, off_id] = ori_data_offense_obs[i,
                                                                         :, :, off_order[i, off_id]]
                data_defense_obs[i, :, :, off_id] = ori_data_defense_obs[i,
                                                                         :, :, off_order[i, off_id]]
                data_act[i, :, off_id] = ori_data_act[i,
                                                      :, off_order[i, off_id]]
                data_init_vel[i, off_id] = ori_data_init_vel[i,
                                                             off_order[i, off_id]]
        assert (data_obs[:, :, :, 1:6] != ori_data_offense_obs).any(
        ), "data_obs(offense) should be modified"
        assert (data_obs[:, :, :, 6:11] != ori_data_defense_obs).any(
        ), "data_obs(defense) should be modified"
        assert (data_act != ori_data_act).any(), "data_act should be modified"
        assert (data_init_vel != ori_data_init_vel).any(
        ), "data_init_vel should be modified"

        # 6. save files into h5py
        with h5py.File('OrderedGAILTransitionData_{}.hdf5'.format(ENV_CONDITION_LENGTH), "w") as f:
            dset = f.create_dataset('OBS', data=data_obs)  # first is useless
            dset = f.create_dataset('DEF_ACT', data=data_act)
            dset = f.create_dataset('DEF_INIT_VEL', data=data_init_vel)
        print('Saved')


if __name__ == "__main__":
    main()
