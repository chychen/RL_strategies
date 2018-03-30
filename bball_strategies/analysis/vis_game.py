import numpy as np
import argparse

import os
from os import listdir
from os.path import join

import time
import matplotlib
matplotlib.use('agg')  # run backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc


def update_all(frame_id, player_circles, ball_circle, annotations, data, vis_ball_height):
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
    if vis_ball_height:
        ball_circle.set_radius(0.4 + data[frame_id, 2] / 10.0)
    else:
        ball_circle.set_radius(0.4)

    annotations[10].set_position(ball_circle.center)
    return


def update_compare_all(frame_id, match_distance_text, player_circles, ball_circle, lines, best_match_pairs, permu_min, annotations, data_1, data_2, vis_ball_height):
    """ 
    Inputs
    ------
    frame_id : int
        automatically increased by 1
    player_circles : list of pyplot.Circle
        players' icon
    ball_circle : list of pyplot.Circle
        ball's icon
    lines : list of Line2D
        linking between data_1 defense and data_2 defense
    best_match_pairs : int, shape=[length, 5]
        for each real defense(data_1), find the best match to fake defense(data_2)
    annotations : pyplot.axes.annotate
        colors, texts, locations for ball and players
    data : float, shape=[length, 23]
        23 = ball's xyz + 10 players's xy
    """
    # text
    match_distance_text.set_text('match/frame: {0:6.4f} ft'.format(permu_min[frame_id]))
    # players
    for j, circle in enumerate(player_circles):
        if j < 10:
            circle.center = data_1[frame_id, 3 + j *
                                   2 + 0], data_1[frame_id, 3 + j * 2 + 1]
            annotations[j].set_position(circle.center)
        else:
            idx = j-10+5  # data_2's defense
            circle.center = data_2[frame_id, 3 + idx *
                                   2 + 0], data_2[frame_id, 3 + idx * 2 + 1]
            annotations[j].set_position(circle.center)
    # lines
    for i in range(5):
        real_idx = i+5
        fake_idx = best_match_pairs[frame_id, i] + 5
        x1 = data_1[frame_id, 3 + real_idx * 2 + 0]
        y1 = data_1[frame_id, 3 + real_idx * 2 + 1]
        x2 = data_2[frame_id, 3 + fake_idx * 2 + 0]
        y2 = data_2[frame_id, 3 + fake_idx * 2 + 1]
        lines[i].set_xdata([x1, x2])
        lines[i].set_ydata([y1, y2])

    # ball
    ball_circle.center = data_1[frame_id, 0], data_1[frame_id, 1]
    if vis_ball_height:
        ball_circle.set_radius(0.4 + data_1[frame_id, 2] / 10.0)
    else:
        ball_circle.set_radius(0.4)

    annotations[15].set_position(ball_circle.center)
    return


def plot_compare_data(data_1, data_2, length, permu_min, best_match_pairs, file_path=None, if_save=False, fps=5, dpi=128, vis_ball_height=False, vis_annotation=False):
    """
    Inputs
    ------
    data_1 : float, shape=[length, 23]
        23 = ball's xyz + 10 players's xy
    data_2 : float, shape=[length, 23]
        23 = ball's xyz + 10 players's xy
    length : int
        how long would you like to plot
    permu_min : float, shape=[length]
    best_match_pairs : int, shape=[length, 5]
        for each real defense(data_1), find the best match to fake defense(data_2)
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
                 'B1', 'B2', 'B3', 'B4', 'B5',
                 'C1', 'C2', 'C3', 'C4', 'C5', 'O']

    # team A -> read circle, team B -> blue circle, ball -> small green circle
    player_circles = []
    [player_circles.append(plt.Circle(xy=(0, 0), radius=1.25, color='r'))
     for _ in range(5)]
    [player_circles.append(plt.Circle(xy=(0, 0), radius=1.25, color='b'))
     for _ in range(5)]
    [player_circles.append(plt.Circle(xy=(0, 0), radius=1.25, color='#000033'))  # grey
     for _ in range(5)]
    ball_circle = plt.Circle(xy=(0, 0), radius=0.4, color='g')

    # lines
    lines = []
    [lines.append(Line2D([0], [0])) for _ in range(5)]

    # text
    match_distance_text = Text(x=0, y=60, text='match/frame: {0:6.4f} ft'.format(0), size=15)
    permu_min = permu_min[:length]
    total_distance_text = Text(x=0, y=52.5, text='mean: {0:6.4f} ft'.format(np.mean(permu_min)), size=15)

    # plot
    ax = plt.axes(xlim=(0, 100), ylim=(-20, 70))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    ax.add_artist(match_distance_text)
    ax.add_artist(total_distance_text)
    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)
    for line in lines:
        ax.add_line(line)

    # annotations on circles
    if vis_annotation:
        annotations = [ax.annotate(name_list[i], xy=[0., 0.],
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for i in range(16)]
    else:
        annotations = [ax.annotate('', xy=[0., 0.],
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for i in range(16)]

    # animation
    anim = animation.FuncAnimation(fig, update_compare_all, fargs=(
        match_distance_text, player_circles, ball_circle, lines, best_match_pairs, permu_min, annotations, data_1, data_2, vis_ball_height), frames=length, interval=200)

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


def plot_data(data, length, file_path=None, if_save=False, fps=5, dpi=128, vis_ball_height=False, vis_annotation=False):
    """
    Inputs
    ------
    data : float, shape=[length, 23]
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
    [player_circles.append(plt.Circle(xy=(0, 0), radius=1.25, color='r'))
     for _ in range(5)]
    [player_circles.append(plt.Circle(xy=(0, 0), radius=1.25, color='b'))
     for _ in range(5)]
    ball_circle = plt.Circle(xy=(0, 0), radius=0.4, color='g')

    # plot
    ax = plt.axes(xlim=(0, 100), ylim=(0, 50))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    # annotations on circles
    if vis_annotation:
        annotations = [ax.annotate(name_list[i], xy=[0., 0.],
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for i in range(11)]
    else:
        annotations = [ax.annotate('', xy=[0., 0.],
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for i in range(11)]

    # animation
    anim = animation.FuncAnimation(fig, update_all, fargs=(
        player_circles, ball_circle, annotations, data, vis_ball_height), frames=length, interval=200)

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

    real_data = np.load('../data/WGAN/FixedFPS5.npy')[:10000:100]
    real_data = np.concatenate([real_data[:, :, 0, :3], real_data[:, :, 1:6, :2].reshape(
        [real_data.shape[0], real_data.shape[1], 10]), real_data[:, :, 6:11, :2].reshape([real_data.shape[0], real_data.shape[1], 10])], axis=-1)
    cnn_wi_data = np.load('../data/WGAN/cnn_wi_2000k/A_fake_B_N100.npy')[0]
    length = np.load('../data/WGAN/FixedFPS5Length.npy')[:10000:100]
    save_path = '../data/WGAN/user_study/fake'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(100):
        try:
            plot_data(cnn_wi_data[i], length=length[i],
                    file_path=save_path+'/play_' + str(i) + '.mp4', if_save=True)
        except:
            pass
    save_path = '../data/WGAN/user_study/real'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(100):
        try:
            plot_data(real_data[i], length=length[i],
                    file_path=save_path+'/play_' + str(i) + '.mp4', if_save=True)
        except:
            pass



    ## cmd e.g. python game_visualizer.py --data_path='../../data/collect/mode_6/results_A_fake_B.npy' --save_path='../../data/collect/try/' --amount=10
    # results_data = np.load('../data/WGAN/FULL.npy')
    # results_len = np.load('../data/WGAN/FULL-LEN.npy')
    # results_data = np.concatenate(
    #     [
    #         # ball
    #         results_data[:, :, 0, :3].reshape(
    #             [results_data.shape[0], results_data.shape[1], 1 * 3]),
    #         # team A players
    #         results_data[:, :, 1:6, :2].reshape(
    #             [results_data.shape[0], results_data.shape[1], 5 * 2]),
    #         # team B players
    #         results_data[:, :, 6:11, :2].reshape(
    #             [results_data.shape[0], results_data.shape[1], 5 * 2])
    #     ], axis=-1
    # )
    # print(results_data.shape)
    # save_path = '../data/WGAN/real'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # for i in range(100):
    #     plot_data(results_data[i], length=results_len[i],
    #               file_path=save_path+'/play_' + str(i) + '.mp4', if_save=True)

    # results_data = np.load('../data/WGAN/results_A_fake_B_wo.npy')
    # print(results_data.shape)
    # save_path = '../data/WGAN/fake_wo'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # for i in range(100):
    #     plot_data(results_data[0, i], length=results_len[i],
    #               file_path=save_path+'/play_' + str(i) + '.mp4', if_save=True)

    # results_data = np.load('../data/WGAN/results_A_fake_B.npy')
    # print(results_data.shape)
    # save_path = '../data/WGAN/fake_wi'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # for i in range(100):
    #     plot_data(results_data[0, i], length=results_len[i],
    #               file_path=save_path+'/play_' + str(i) + '.mp4', if_save=True)

    # exit()

    # target_data = np.zeros(shape=[100,23])
    # off_data = np.load('data/def_obs.npy')
    # target_data[:,0:2] = off_data[:100, 0, -1, 0:1, :].reshape([100,2])
    # target_data[:,3:23] = off_data[:100, 0, -1, 1:11, :].reshape([100,20])
    # plot_data(target_data, length=100,
    #             file_path=opt.save_path + 'play_def.mp4', if_save=opt.save)

    # train_data = np.load(opt.data_path)
    # train_data_len = np.load('../data/FixedFPS5Length.npy')
    # print(train_data.shape)
    # print(train_data_len.shape)
    # target_data = np.concatenate([train_data[:, :, 0:1, :3].reshape(
    #     [train_data.shape[0], 235, 3]), train_data[:, :, 1:, :2].reshape([train_data.shape[0], 235, 20])], axis=-1)
    # for i in range(100):
    #     plot_data(target_data[i], length=train_data_len[i],
    #               file_path=opt.save_path + 'play_{}.mp4'.format(i), if_save=opt.save)
    # transition_idx = 0
    # for i in range(100):
    #     # discard both the head and tail while transform real positions to transitions
    #     transition_idx += (train_data_len[i]-2)
    #     plot_data(target_data[i], length=train_data_len[i],
    #               file_path=opt.save_path + 'play_tran_{}_epi_{}.mp4'.format(transition_idx, i), if_save=opt.save)

    # print('opt.save', opt.save)
    # print('opt.amount', opt.amount)
    # print('opt.seq_length', opt.seq_length)
    # print('opt.save_path', opt.save_path)


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
