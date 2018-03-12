import numpy as np
import argparse

import os
from os import listdir
from os.path import join

import time
import matplotlib
matplotlib.use('agg')  # run backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc


def update_all(frame_id, player_circles, ball_circle, annotations, data):
    """ 
    Inputs
    ------
    frame_id : int
        automatically increased by 1
    player_circles : list of pyplot.Circle
        players' icon
    ball_circle : list of pyplot.Circle
        ball's icon
    annotations : pyplot.axes.annotate
        colors, texts, locations for ball and players
    data : float, shape=[length, 23]
        23 = ball's xyz + 10 players's xy
    """
    # players
    for j, circle in enumerate(player_circles):
        circle.center = data[frame_id, 3 + j *
                             2 + 0], data[frame_id, 3 + j * 2 + 1]
        annotations[j].set_position(circle.center)
    # ball
    ball_circle.center = data[frame_id, 0], data[frame_id, 1]
    ball_circle.set_radius(1.0 + data[frame_id, 2] / 10.0)
    annotations[10].set_position(ball_circle.center)
    return


def plot_data(data, length, file_path=None, if_save=False, fps=6, dpi=128):
    """
    Inputs
    ------
    data : float, shape=[amount, length, 23]
        23 = ball's xyz + 10 players's xy
    length : int
        how long would you like to plot
    file_path : str
        where to save the animation
    if_save : bool, optional
        save as .gif file or not
    fps : int, optional
        frame per second
    dpi : int, optional
        dot per inch
    Return
    ------
    """
    court = plt.imread("../gym_bball/envs/fullcourt.png")  # 500*939
    name_list = ['A1', 'A2', 'A3', 'A4', 'A5',
                 'B1', 'B2', 'B3', 'B4', 'B5', 'O']

    # team A -> read circle, team B -> blue circle, ball -> small green circle
    player_circles = []
    [player_circles.append(plt.Circle(xy=(0, 0), radius=2.5, color='r'))
     for _ in range(5)]
    [player_circles.append(plt.Circle(xy=(0, 0), radius=2.5, color='b'))
     for _ in range(5)]
    ball_circle = plt.Circle(xy=(0, 0), radius=1.5, color='g')

    # plot
    ax = plt.axes(xlim=(0, 100), ylim=(0, 50))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    # annotations on circles
    annotations = [ax.annotate(name_list[i], xy=[0., 0.],
                               horizontalalignment='center',
                               verticalalignment='center', fontweight='bold')
                   for i in range(11)]
    # animation
    anim = animation.FuncAnimation(fig, update_all, fargs=(
        player_circles, ball_circle, annotations, data), frames=length, interval=200)

    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    if if_save:
        if file_path[-3:] == 'mp4':
            anim.save(file_path, fps=fps,
                      dpi=dpi, writer='ffmpeg')
        else:
            anim.save(file_path, fps=fps,
                      dpi=dpi, writer='imagemagick')
        print('!!!Animation is saved!!!')
    else:
        plt.show()
        print('!!!End!!!')

    # clear content
    plt.cla()
    plt.clf()


def test():
    """
    plot real data
    """
    # target_data = np.zeros(shape=[100,23])
    # off_data = np.load('data/def_obs.npy')
    # target_data[:,0:2] = off_data[:100, 0, -1, 0:1, :].reshape([100,2])
    # target_data[:,3:23] = off_data[:100, 0, -1, 1:11, :].reshape([100,20])
    # plot_data(target_data, length=100,
    #             file_path=opt.save_path + 'play_def.mp4', if_save=opt.save)

    train_data = np.load(opt.data_path)
    train_data_len = np.load('../data/FixedFPS5Length.npy')
    print(train_data.shape)
    print(train_data_len.shape)
    target_data = np.concatenate([train_data[:, :, 0:1, :3].reshape(
        [train_data.shape[0], 235, 3]), train_data[:, :, 1:, :2].reshape([train_data.shape[0], 235, 20])], axis=-1)
    # for i in range(100):
    #     plot_data(target_data[i], length=train_data_len[i],
    #               file_path=opt.save_path + 'play_{}.mp4'.format(i), if_save=opt.save)
    transition_idx = 0
    for i in range(100):
        # discard both the head and tail while transform real positions to transitions
        transition_idx += (train_data_len[i]-2)
        plot_data(target_data[i], length=train_data_len[i],
                  file_path=opt.save_path + 'play_tran_{}_epi_{}.mp4'.format(transition_idx, i), if_save=opt.save)

    print('opt.save', opt.save)
    print('opt.amount', opt.amount)
    print('opt.seq_length', opt.seq_length)
    print('opt.save_path', opt.save_path)


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser(description='NBA Games visulization')
    parser.add_argument('--save', type=bool, default=True,
                        help='bool, if save as gif file')
    parser.add_argument('--amount', type=int, default=100,
                        help='how many event do you want to plot')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='how long for each event')
    parser.add_argument('--save_path', type=str, default='../data/',
                        help='string, path to save event animation')
    parser.add_argument('--data_path', type=str,
                        default='../data/FixedFPS5.npy', help='string, path of target data')

    opt = parser.parse_args()
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    test()