""" Translate real sequencial data to state-action dataset for policy supervised pretraining.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np

TEAM_LIST = ['OFFENSE', 'DEFENSE']
STATUS_LIST = ['SHOT', 'PASSED', 'DRIBBLED', 'CAUGHT', 'FLYING']

DECISION_LOOKUP = {
    'SHOOT': 0,
    'PASS': 1,
    'NO_OP': 2
}

STATUS_LOOKUP = {
    'SHOT': DECISION_LOOKUP['SHOOT'],
    'PASSED': DECISION_LOOKUP['PASS'],
    'DRIBBLED': DECISION_LOOKUP['NO_OP'],
    'CAUGHT': DECISION_LOOKUP['NO_OP'],
    'FLYING': DECISION_LOOKUP['NO_OP']
}

RIGHT_BASKET_POS = [94 - 5.25, 25]
WINGSPAN_RADIUS = 3.5
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
BUFFER_SIZE = 5


def length(vec, axis=1):
    return np.sqrt(np.sum(vec * vec, axis=axis))


def get_ball_status(last_ball_pos, ball_pos, next_ball_pos, last_off_pos, off_pos, next_off_pos):
    """
    Inputs
    ------
    last_ball_pos, ball_pos, next_ball_pos : float, shape=[1,2]
    last_off_pos, off_pos, next_off_pos : float, shape=[5,2]

    Returns
    -------
    status : str
        'SHOT' or 'PASSED' or 'DRIBBLED' or 'CAUGHT' or 'FLYING'
    ball_dir : float
        if status=='DRIBBLED' or 'CAUGHT' or 'FLYING' -> return [.0, .0],
        else -> return direction
    ball_handler_idx : 
        if status=='DRIBBLED' or 'CAUGHT' -> return int,
        else -> return None
    """
    status = str()
    ball_dir = [.0, .0]
    ball_handler_idx = None

    cur_len = np.amin(length(ball_pos - off_pos, axis=1))
    ball_handler_idx = np.argmin(length(ball_pos - off_pos, axis=1))
    last_len = length(last_ball_pos - last_off_pos, axis=1)[ball_handler_idx]
    next_len = length(next_ball_pos - next_off_pos, axis=1)[ball_handler_idx]

    judge = [last_len < WINGSPAN_RADIUS, cur_len <
             WINGSPAN_RADIUS, next_len < WINGSPAN_RADIUS]
    if judge == [True, True, False]:
        ball_vel = next_ball_pos[0] - ball_pos[0]
        ball2basket_vec = RIGHT_BASKET_POS - ball_pos[0]
        dot_value = np.dot(ball_vel, ball2basket_vec)
        if_angle_small = np.arccos(
            dot_value / (length(ball_vel, axis=0) * length(ball2basket_vec, axis=0) + 1e-8)) < np.pi * 3.0 / 180.0 # prevent from dividing 0
        if dot_value > 0 and if_angle_small:
            status = 'SHOT'
        else:
            status = 'PASSED'
        ball_dir = ball_vel / (length(ball_vel, axis=0) + 1e-8) # prevent from dividing 0
    elif judge == [True, True, True] or judge == [False, True, True]:
        status = 'DRIBBLED'
        ball_handler_idx = np.argmin(length(ball_pos - off_pos, axis=1))
    elif judge == [False, False, True]:
        status = 'CAUGHT'
        ball_handler_idx = np.argmin(length(ball_pos - off_pos, axis=1))
    elif judge == [False, False, False] or judge == [True, False, False]:
        status = 'FLYING'
    # NOTE not 100% sure
    elif judge == [True, False, True]:
        status = 'CAUGHT'
        ball_handler_idx = np.argmin(length(ball_pos - off_pos, axis=1))
    # NOTE not 100% sure
    elif judge == [False, True, False]:
        status = 'FLYING'
    else:
        # not defined : [True, False, True], [False, True, False]
        print(length(last_ball_pos - last_off_pos, axis=1))
        print(length(ball_pos - off_pos, axis=1))
        print(length(next_ball_pos - next_off_pos, axis=1))
        msg = 'Ball status is not defined, with judge: {}'.format(judge)
        raise ValueError(msg)

    return status, ball_dir, ball_handler_idx


def packing_data(data, data_len, mode):
    """
    Inputs
    ------
    data : float, shape=[num_episode, max_length, 11, 4]
        11 -> [ball, 5 offense, 5 defense]
        4 -> [x, y, z, player position]
    data_len : int, shape=[num_episode,]
        true length of each episode
    mode : str, 'OFFENSE' or 'DEFENSE'

    Returns
    -------
    obs : float, shape=[num_episode, max_length-2, BUFFER_SIZE, 14, 2]
    actions : float
        if mode=='OFFENSE' -> shape=[num_episode, max_length-2, 15]
        if mode=='DEFENSE' -> shape=[num_episode, max_length-2, 10]
    """
    assert mode in TEAM_LIST, "mode must be eigther offense or defense"
    obs = np.empty(shape=[data.shape[0], data.shape[1]-2, 5, 14, 2])
    obs = []
    action_shape = 15 if mode == 'OFFENSE' else 10
    actions = np.empty(shape=[data.shape[0], data.shape[1]-2, action_shape])
    actions = []
    for epi_idx in range(data.shape[0]):
        # print(epi_idx)
        # obs buffer, filled up with 5 first frame
        obs_buffer = np.empty(shape=[BUFFER_SIZE, 14, 2])
        for i in range(BUFFER_SIZE):
            obs_buffer[i] = np.concatenate([
                data[epi_idx, 0, :, :2],
                np.expand_dims(RIGHT_BASKET_POS, axis=0),
                np.expand_dims([COURT_LENGTH / 2, 0], axis=0),
                np.expand_dims([COURT_LENGTH, COURT_WIDTH], axis=0)
            ], axis=0)

        # make sure their exist last and next value
        for len_idx in range(1, data_len[epi_idx]-1):
            # last
            last_ball_pos = data[epi_idx, len_idx-1, 0:1, :2]
            last_off_pos = data[epi_idx, len_idx-1, 1:6, :2]
            last_def_pos = data[epi_idx, len_idx-1, 6:11, :2]
            # current
            ball_pos = data[epi_idx, len_idx, 0:1, :2]
            off_pos = data[epi_idx, len_idx, 1:6, :2]
            def_pos = data[epi_idx, len_idx, 6:11, :2]
            # next
            next_ball_pos = data[epi_idx, len_idx+1, 0:1, :2]
            next_off_pos = data[epi_idx, len_idx+1, 1:6, :2]
            next_def_pos = data[epi_idx, len_idx+1, 6:11, :2]
            # Vel = Pos - Last_Pos
            ball_vel = ball_pos - last_ball_pos
            off_vel = off_pos - last_off_pos
            def_vel = def_pos - last_def_pos
            # Next_Vel = Next_Pos - Pos
            next_ball_vel = next_ball_pos - ball_pos
            next_off_vel = next_off_pos - off_pos
            next_def_vel = next_def_pos - def_pos
            # Acc = Next_Vel - Vel
            off_acc = next_off_vel - off_vel
            def_acc = next_def_vel - def_vel

            ret_len_idx = len_idx-1
            if mode == 'OFFENSE':
                temp_act = np.empty([15, ])
                # check ball status, [dribbled or passed, shot]
                ball_status, ball_dir, ball_handler_idx = get_ball_status(
                    last_ball_pos, ball_pos, next_ball_pos, last_off_pos, off_pos, next_off_pos)
                assert ball_status in STATUS_LIST
                # action's decision, (one hot vec)
                for _, value in DECISION_LOOKUP.items():
                    temp_act[value] = float(
                        STATUS_LOOKUP[ball_status] == value)
                # ball's direction
                temp_act[3:5] = ball_dir
                # if 'DRIBBLED' or 'CAUGHT', assign the ball handler position to ball position
                if ball_status == 'DRIBBLED' or ball_status == 'CAUGHT':
                    assert ball_handler_idx is not None
                    ball_pos = off_pos[ball_handler_idx]
                # action's dash
                temp_act[5:] = np.reshape(off_acc, [10])
                actions.append(copy.deepcopy(temp_act))
            elif mode == 'DEFENSE':
                # action's dash
                actions.append(copy.deepcopy(np.reshape(off_acc, [10])))
            # pack the return data
            obs_buffer[:BUFFER_SIZE-1] = copy.deepcopy(obs_buffer[1:])
            obs_buffer[-1] = np.concatenate([
                data[epi_idx, len_idx, :, :2],
                np.expand_dims(RIGHT_BASKET_POS, axis=0),
                np.expand_dims([COURT_LENGTH / 2, 0], axis=0),
                np.expand_dims([COURT_LENGTH, COURT_WIDTH], axis=0)
            ], axis=0)
            obs.append(copy.deepcopy(obs_buffer))

    obs = np.stack(obs, axis=0)
    actions = np.stack(actions, axis=0)
    return obs, actions


def state_2_action(data, data_len):
    """ 
    Inputs
    ------
    data : float, shape=[num_episode, max_length, 11, 4]
        11 -> [ball, 5 offense, 5 defense]
        4 -> [x, y, z, player position]
    data_len : int, shape=[num_episode,]
        true length of each episode

    Return
    ------
    off_obs: float, shape=[num_episode*true_length, 5, 14, 2]
        timesteps of [off, def] = [t-1, t-1]
        shape is as same as the return value of env.etep()
        5 -> buffer 5 frames
        14 -> [1 ball, 5 offense, 5 defense, 1 basket pos, 2 boundary info]
    def_obs: float, shape=[num_episode*true_length, 5, 14, 2]
        timesteps of [off, def] = [t, t-1]
        shape is as same as the return value of env.etep()
        5 -> buffer 5 frames
        14 -> [1 ball, 5 offense, 5 defense, 1 basket pos, 2 boundary info]
    off_actions : float, shape=[num_episode*true_length, 15]
        shape is as same as the output value of the offense pretrain network
        15 -> [3 one hot decision, 2 ball, 5*2 offense]
    def_actions : float, shape=[num_episode*true_length, 10]
        shape is as same as the output value of the defense pretrain network
        10 -> [5*2 defense]
    """
    # packing offense training data
    off_obs, off_actions = packing_data(data, data_len, mode='OFFENSE')

    # packing defense training data
    # in the obs of defense, offense always one step lead than defense
    data[:, :-1, 1:6] = data[:, 1:, 1:6]
    def_obs, def_actions = packing_data(data, data_len, mode='DEFENSE')

    return off_obs, def_obs, off_actions, def_actions


def main():
    # 1. data IO
    data = np.load('../../data/FPS5.npy')
    data_len = np.load('../../data/FPS5Length.npy')
    print(data.shape)
    print(data_len.shape)
    # 2. state_2_action : shoot, pass, dash
    off_obs, def_obs, off_actions, def_actions = state_2_action(data, data_len)
    # 3. save as numpy
    np.save('off_obs.npy', off_obs[:, None])
    np.save('def_obs.npy', def_obs[:, None])
    np.save('off_actions.npy', off_actions[:, None])
    np.save('def_actions.npy', def_actions[:, None])
    print('saved complete!!')


if __name__ == '__main__':
    main()
