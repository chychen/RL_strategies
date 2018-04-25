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
WINGSPAN_RADIUS = 3.5 + 0.5
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
BUFFER_SIZE = 5
SHOOT_ANGLE = 3.0
SHOOT_LEFT_FRAMES = 15


def length(vec, axis=1):
    return np.sqrt(np.sum(vec * vec, axis=axis))


def get_ball_status(ball_pos, off_pos, frames_left):
    """
    Inputs
    ------
    ball_pos : float, shape=[4,1,2]
        [t-1, t, t+1, t+2]
    off_pos : float, shape=[4,5,2]
        [t-1, t, t+1, t+2]
    frames_left : int, shape=()
        left frames to the episode's end

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

    Ball Status Note
    ----------------
    [t-1, t, t+1, t+2] : if ball in WINGSPAN_RADIUS

    [True, True, True, True] -> DRIBBLED
    [True, True, True, False] -> DRIBBLED : pass ball in near feature
    [True, True, False, True] -> None : imposiible : DRIBBLED
    [True, True, False, False] -> SHOT or PASSED

    [True, False, True, True] -> CAUGHT : maybe...
    [True, False, True, False] -> None : imposiible : DRIBBLED
    [True, False, False, True] -> FLYING : catch ball in near feature
    [True, False, False, False] -> FLYING

    [False, True, True, True] -> DRIBBLED
    [False, True, True, False] -> DRIBBLED : maybe...
    [False, True, False, True] -> None : imposiible : DRIBBLED
    [False, True, False, False] -> None : imposiible : DRIBBLED

    [False, False, True, True] -> CAUGHT
    [False, False, True, False] -> None : imposiible : DRIBBLED
    [False, False, False, True] -> FLYING : catch ball in near feature
    [False, False, False, False] -> FLYING
    """
    status = str()
    ball_dir = [.0, .0]
    ball_handler_idx = None

    ball_handler_idx = np.argmin(length(ball_pos[1] - off_pos[1], axis=1))
    len_ = length(ball_pos - off_pos, axis=2)[:, ball_handler_idx]
    judge = [len_[0] < WINGSPAN_RADIUS, len_[1] <
             WINGSPAN_RADIUS, len_[2] < WINGSPAN_RADIUS, len_[3] < WINGSPAN_RADIUS]

    if judge == [True, True, False, False]:
        # because the simulated env setting, actions of ball should be after dash, so we analysis ball status of 't+1' instead of 't'
        next_ball_vel = ball_pos[3, 0] - ball_pos[2, 0]
        ball2basket_vec = RIGHT_BASKET_POS - ball_pos[2, 0]
        dot_value = np.dot(next_ball_vel, ball2basket_vec)
        if_angle_small = np.arccos(
            dot_value / (length(next_ball_vel, axis=0) * length(ball2basket_vec, axis=0) + 1e-8)) < np.pi * SHOOT_ANGLE / 180.0  # prevent from dividing 0
        if dot_value > 0 and if_angle_small and frames_left <= SHOOT_LEFT_FRAMES:
            status = 'SHOT'
        else:
            status = 'PASSED'
        # prevent from dividing 0
        ball_dir = next_ball_vel / (length(next_ball_vel, axis=0) + 1e-8)
        ball_handler_idx = None
    elif judge == [True, True, True, True] or judge == [True, True, True, False] or judge == [False, True, True, True] or judge == [False, True, True, False]:
        status = 'DRIBBLED'
    elif judge == [True, False, True, True] or judge == [False, False, True, True]:
        status = 'CAUGHT'
    elif judge == [True, False, False, True] or judge == [True, False, False, False] or judge == [False, False, False, True] or judge == [False, False, False, False]:
        status = 'FLYING'
        ball_handler_idx = None
    else:
        status = 'DRIBBLED'
        # print(length(ball_pos - off_pos, axis=2))
        # msg = 'Ball status is not defined, with judge: {}'.format(judge)
        # raise ValueError(msg)

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
    num_of_transitions = sum over (num_episode) with (true_length-3)
    obs : float, shape=[num_of_transitions, BUFFER_SIZE, 14, 2]
    actions : float
        if mode=='OFFENSE' -> shape=[num_of_transitions, 15]
        if mode=='DEFENSE' -> shape=[num_of_transitions, 10]
    """
    assert mode in TEAM_LIST, "mode must be eigther offense or defense"
    obs = np.empty(shape=[data.shape[0], data.shape[1]-3, 5, 14, 2])
    obs = []
    action_shape = 15 if mode == 'OFFENSE' else 10
    actions = []
    for epi_idx in range(data.shape[0]):
        if epi_idx % 100 == 0:
            print('{}/{}'.format(epi_idx, data.shape[0]))
        # obs buffer, filled up with 5 first frame
        obs_buffer = np.empty(shape=[BUFFER_SIZE, 14, 2])
        for i in range(BUFFER_SIZE):
            obs_buffer[i] = np.concatenate([
                data[epi_idx, 0, :, :2],
                np.expand_dims(RIGHT_BASKET_POS, axis=0),
                np.expand_dims([COURT_LENGTH / 2, 0], axis=0),
                np.expand_dims([COURT_LENGTH, COURT_WIDTH], axis=0)
            ], axis=0)
        # judge actions by checking the latest 4 frames [t-1, t, t+1, t+2]
        # make sure their exist last and next 2 value
        for len_idx in range(1, data_len[epi_idx]-2):
            # pos
            ball_pos = data[epi_idx, len_idx-1:len_idx+3, 0:1, :2]
            off_pos = data[epi_idx, len_idx-1:len_idx+3, 1:6, :2]
            def_pos = data[epi_idx, len_idx-1:len_idx+3, 6:11, :2]
            # vel
            ball_vel = ball_pos[1:] - ball_pos[:-1]
            off_vel = off_pos[1:] - off_pos[:-1]
            def_vel = def_pos[1:] - def_pos[:-1]
            # Acc = Next_Vel - Vel
            off_acc = off_vel[1:] - off_vel[:-1]
            def_acc = def_vel[1:] - def_vel[:-1]

            if mode == 'OFFENSE':
                temp_act = np.empty([15, ])
                # check ball status, [dribbled or passed, shot]
                ball_status, ball_dir, ball_handler_idx = get_ball_status(
                    ball_pos, off_pos, data_len[epi_idx]-len_idx)
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
                    ball_pos[1] = off_pos[1][ball_handler_idx]
                # action's dash
                temp_act[5:] = np.reshape(off_acc[0], [10])
                actions.append(copy.deepcopy(temp_act))
            elif mode == 'DEFENSE':
                # action's dash
                actions.append(copy.deepcopy(np.reshape(def_acc[0], [10])))
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
    num_of_transitions = sum over (num_episode) with (true_length-3)
    off_obs: float, shape=[num_of_transitions, 5, 14, 2]
        timesteps of [off, def] = [t-1, t-1]
        shape is as same as the return value of env.etep()
        5 -> buffer 5 frames
        14 -> [1 ball, 5 offense, 5 defense, 1 basket pos, 2 boundary info]
    def_obs: float, shape=[num_of_transitions, 5, 14, 2]
        timesteps of [off, def] = [t, t-1]
        shape is as same as the return value of env.etep()
        5 -> buffer 5 frames
        14 -> [1 ball, 5 offense, 5 defense, 1 basket pos, 2 boundary info]
    off_actions : float, shape=[num_of_transitions, 15]
        shape is as same as the output value of the offense pretrain network
        15 -> [3 one hot decision, 2 ball, 5*2 offense]
    def_actions : float, shape=[num_of_transitions, 10]
        shape is as same as the output value of the defense pretrain network
        10 -> [5*2 defense]
    """
    # packing offense training data
    off_obs, off_actions = packing_data(data, data_len, mode='OFFENSE')
    print('OFFENSE is ready')

    # packing defense training data
    # in the obs of defense, offense always one step lead than defense
    data[:, :-1, 1:6] = data[:, 1:, 1:6]
    def_obs, def_actions = packing_data(data, data_len, mode='DEFENSE')
    print('DEFENSE is ready')

    return off_obs, def_obs, off_actions, def_actions


def main():
    # 1. data IO
    data = np.load('../../data/FixedFPS5.npy')
    data_len = np.load('../../data/FixedFPS5Length.npy')
    print(data.shape)
    print(data_len.shape)
    # 2. state_2_action : shoot, pass, dash
    off_obs, def_obs, off_actions, def_actions = state_2_action(data, data_len)
    # 3. save as numpy
    np.save('off_obs.npy', off_obs[:, None])
    np.save('def_obs.npy', def_obs[:, None])
    np.save('off_actions.npy', off_actions[:, None])
    np.save('def_actions.npy', def_actions[:, None])
    print('off_obs shape', off_obs.shape)
    print('def_obs shape', def_obs.shape)
    print('off_actions shape', off_actions.shape)
    print('def_actions shape', def_actions.shape)
    print('saved complete!!')


if __name__ == '__main__':
    main()
