import numpy as np
import argparse
from operator import itemgetter
import os
from os import listdir
from os.path import join
from PIL import Image
# from tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc
import random


def update_all(i, player_circles, ball_circle, annotations, oh, factor):
    for j, circle in enumerate(player_circles):
        # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
        circle.center = oh[0, i , j * 2 +3] * factor, oh[0, i , j * 2 +4] * factor
        annotations[j].set_position(circle.center)
    
    ball_circle.center = oh[0, i , 0] * factor, oh[0, i, 1] * factor
    ball_circle.set_radius(16. /1.7 + (oh[0, i, 2] - 1.7) * 0.7 )
    annotations[10].set_position(ball_circle.center)
    return player_circles, ball_circle




def SingleSide(data, save_path, c_index=None, n_index=None, n_frames=100):

    go = data
    factor = 8
    # one event
    oh = go[0, :, :]
    name = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'O']
    color = ['black' for _ in range(11)]
    ax = plt.axes(xlim=(0, 94 * factor), ylim=(0,50 * factor + 30))

    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    player_circles = [plt.Circle((0,0), 15, color='#DC143C') for i in range(5)]
    player_circles += [plt.Circle((0,0), 15, color='#87CEEB') for i in range(5)]

    ball_circle = plt.Circle((0,0), 12, color='g')

    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    annotations = [ax.annotate(name[i], xy=[0, 0], color=color[i],
                    horizontalalignment='center',
                    verticalalignment='center', fontweight='bold', fontsize=10)
                    for i in range(11)]
    ax.annotate("CNN model trained at 0 K iterations", xy=[30, 50 * factor + 20], color='black', verticalalignment='center', fontweight='bold', fontsize=10)
    ax.annotate("Total frames : {} ".format(n_frames) , xy=[530, 50 * factor + 20], color='black', verticalalignment='center', fontweight='bold', fontsize=10)

    anim = animation.FuncAnimation(fig, update_all, fargs=(player_circles, ball_circle, annotations, go, factor), frames=n_frames, interval=100)
    
    court = plt.imread("full.png")
    plt.imshow(court, zorder=0, extent=[0, 94 * factor, 50 * factor, 0])
    #if save:
    
    anim.save(save_path, dpi=150, fps=6, writer='ffmpeg')
    plt.close(fig)
    print("Saved ", save_path)
    # plt.show()
    # print(go.shape)



def update_both(i, player_circles_left, ball_circle_left, player_circles_right, ball_circle_right, annotations, data_left, data_right, factor):
    for j, circle in enumerate(player_circles_left):
        # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
        circle.center = data_left[0, i , j * 2 +3] * factor, data_left[0, i , j * 2 +4] * factor
        annotations[j].set_position(circle.center)
    
    ball_circle_left.center = data_left[0, i , 0] * factor, data_left[0, i, 1] * factor
    ball_circle_left.set_radius(16. /1.7 + (data_left[0, i, 2] - 1.7) * 0.7 )
    annotations[10].set_position(ball_circle_left.center)

    offset = 50 * factor + 40
    for j, circle in enumerate(player_circles_right):
        # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
        circle.center = data_right[0, i , j * 2 +3] * factor , data_right[0, i , j * 2 +4] * factor + offset
        annotations[j + 11].set_position(circle.center)
    
    ball_circle_right.center = data_right[0, i , 0] * factor, data_right[0, i, 1] * factor + offset
    ball_circle_right.set_radius(16. / 1.7 + (data_left[0, i, 2] - 1.7) * 0.7 )
    annotations[21].set_position(ball_circle_right.center)



    return player_circles_left, ball_circle_left, player_circles_right, ball_circle_right

def update_bothSpecial(i, player_circles_left, ball_circle_left, player_circles_right, ball_circle_right, annotations, data_left, data_right, factor):
    offset1 = 0
    for j, circle in enumerate(player_circles_left):
        # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
        circle.center = data_left[0, i , j * 2 +3] * factor + offset1, data_left[0, i , j * 2 +4] * factor
        annotations[j].set_position(circle.center)
    
    ball_circle_left.center = data_left[0, i , 0] * factor + offset1, data_left[0, i, 1] * factor
    ball_circle_left.set_radius(16. /1.7 + (data_left[0, i, 2] - 1.7) * 0.7 )
    annotations[10].set_position(ball_circle_left.center)

    offset = 50 * factor + 40
    for j, circle in enumerate(player_circles_right):
        # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
        circle.center = data_right[0, i , j * 2 +3] * factor + offset1, data_right[0, i , j * 2 +4] * factor + offset
        annotations[j + 11].set_position(circle.center)
    
    ball_circle_right.center = data_right[0, i , 0] * factor + offset1, data_right[0, i, 1] * factor + offset
    ball_circle_right.set_radius(16. / 1.7 + (data_left[0, i, 2] - 1.7) * 0.7 )
    annotations[21].set_position(ball_circle_right.center)



    return player_circles_left, ball_circle_left, player_circles_right, ball_circle_right



def BothSide(data_left, data_right, save_path, c_index):
    
    factor = 8
    # one event
    print(data_left.shape, data_right.shape)
    name = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'O', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'O']
    color = ['black' for _ in range(22)]
    ##offset = 100
    ax = plt.axes(xlim=(0, 94 * factor), ylim=(0,50 * factor * 2 + 90))
    print("length", len(name), len(color))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    player_circles_left = [plt.Circle((0,0), 15, color='#DC143C') for i in range(5)] + [plt.Circle((0,0), 15, color='#87CEEB') for i in range(5)]
    ball_circle_left = plt.Circle((0,0), 12, color='g')

    player_circles_right = [plt.Circle((0,0), 15, color='#DC143C') for i in range(5)] + [plt.Circle((0,0), 15, color='#87CEEB') for i in range(5)]
    ball_circle_right = plt.Circle((0,0), 12, color='g')
    # ax.annotate("Example {} ".format(c_index) , xy=[30, 50 * factor * 2 + 70], color='black', verticalalignment='center', fontweight='bold', fontsize=10)
    ax.annotate("Example in Fig. 2 " , xy=[30, 50 * factor * 2 + 70], color='black', verticalalignment='center', fontweight='bold', fontsize=10)
    
    ax.annotate("UP: RNN  Down: CNN" , xy=[230, 50 * factor + 20], color='black', verticalalignment='center', fontweight='bold', fontsize=10)

    for circle in player_circles_left:
        ax.add_patch(circle)
    ax.add_patch(ball_circle_left)

    for circle in player_circles_right:
        ax.add_patch(circle)
    ax.add_patch(ball_circle_right)

    annotations = [ax.annotate(name[i], xy=[0, 0], color=color[i],
                    horizontalalignment='center',
                    verticalalignment='center', fontweight='bold',fontsize=6)
                    for i in range(22)]

    anim = animation.FuncAnimation(fig, update_both, fargs=(player_circles_left, ball_circle_left, player_circles_right, ball_circle_right, annotations, data_left, data_right, factor), frames=100, interval=100)
    
    court = plt.imread("bothfull.png")
    plt.imshow(court, zorder=0, extent=[0, 94 * factor, 50 * factor* 2 + 40, 0])
    #if save:
    
    anim.save(save_path, dpi=200,fps=6, writer='ffmpeg')
    plt.close(fig)
    print("Saved ", save_path)

def BothSideSpecial(data_left, data_right, save_path, c_index):
    
    factor = 8
    # one event
    print(data_left.shape, data_right.shape)
    name = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'O', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'O']
    color = ['black' for _ in range(22)]
    offset = 0
    ##origin is 100
    ax = plt.axes(xlim=(0, 94 * factor + offset), ylim=(0,50 * factor * 2 + 90))
    print("length", len(name), len(color))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    player_circles_left = [plt.Circle((0,0), 15, color='#DC143C') for i in range(5)] + [plt.Circle((0,0), 15, color='#87CEEB') for i in range(5)]
    ball_circle_left = plt.Circle((0,0), 12, color='g')

    player_circles_right = [plt.Circle((0,0), 15, color='#DC143C') for i in range(5)] + [plt.Circle((0,0), 15, color='#87CEEB') for i in range(5)]
    ball_circle_right = plt.Circle((0,0), 12, color='g')
    # ax.annotate("Example {} ".format(c_index) , xy=[30, 50 * factor * 2 + 70], color='black', verticalalignment='center', fontweight='bold', fontsize=10)
    ax.annotate("Example 1." , xy=[offset + 30, 50 * factor * 2 + 70], color='black', verticalalignment='center', fontweight='bold', fontsize=10)
    
    # ax.annotate("CNN" , xy=[0, 50 * factor // 2  ], color='black', verticalalignment='center', fontweight='bold', fontsize=8)
    # ax.annotate("RNN" , xy=[0, 50 * factor // 2 * 3 + 20], color='black', verticalalignment='center', fontweight='bold', fontsize=8)
    # ax.annotate("without wide open penalty" , xy=[0, 50 * factor // 2 * 3], color='black', verticalalignment='center', fontweight='bold', fontsize=6)
    
    for circle in player_circles_left:
        ax.add_patch(circle)
    ax.add_patch(ball_circle_left)

    for circle in player_circles_right:
        ax.add_patch(circle)
    ax.add_patch(ball_circle_right)

    annotations = [ax.annotate(name[i], xy=[0, 0], color=color[i],
                    horizontalalignment='center',
                    verticalalignment='center', fontweight='bold',fontsize=6)
                    for i in range(22)]

    anim = animation.FuncAnimation(fig, update_bothSpecial, fargs=(player_circles_left, ball_circle_left, player_circles_right, ball_circle_right, annotations, data_left, data_right, factor), frames=100, interval=100)
    
    court = plt.imread("bothfull.png")
    plt.imshow(court, zorder=0, extent=[offset, 94 * factor + offset, 50 * factor* 2 + 40, 0])
    #if save:
    
    anim.save(save_path, dpi=200,fps=6, writer='ffmpeg')
    plt.close(fig)
    print("Saved ", save_path)

def Visualization(mode=1, rf_path=None, rr_path=None, sc1_path=None, sc2_path=None, st_path=None, select_list=None, length_list=None, name_list=None):
    """
    """
    if mode == 1:
        #down is real ,up is fake if M == 1
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        B = np.load(rr_path)
        print("A shape ", A.shape)
        print("B shape ", B.shape)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 10
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)
        c_list = []

        score_list = [((i, j), scores[j, i])  for j in range(noise_size) for i in range(condition_size)]
        score_list.sort(key=itemgetter(1), reverse=True)
        ##Denote Fake VS Real flipping
        mark = 0
        for idx in range(condition_size * noise_size):
            c_idx, n_idx = score_list[idx][0]
            if c_idx in c_list:
                continue
            else:
                c_list.append(c_idx)
            score = score_list[idx][1]
            up_pos = np.expand_dims(A[n_idx, c_idx, :, :], axis=0)
            down_pos = np.expand_dims(B[c_idx, :, :], axis=0)
            save_name = "{}_S{}_M{}_C{}_N{}.mp4".format(idx, score, mark, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            if mark == 0:
                BothSide(up_pos, down_pos, save_path)
                mark = 1
            else:
                BothSide(down_pos, up_pos, save_path)
                mark = 0
            if len(c_list) > condition_range:
                break
    elif mode == 2:
        scores = np.load(sc1_path)
        print("Score shape", scores.shape)
        A = np.load(rf_path)
        print("RF shape ", A.shape)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 10
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)
        score_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        allsort_list = [((i, j), scores[j, i])  for j in range(noise_size) for i in range(condition_size)]
        for each in score_list:
            each.sort(key=itemgetter(1), reverse=True)
        allsort_list.sort(key=itemgetter(1), reverse=True)
        c_list = []
        for i in range(condition_size * noise_size):
            c_idx, n_idx = allsort_list[i][0]
            if c_idx not in c_list:
                c_list.append(c_idx)
            else:
                continue
            if len(c_list) > condition_range:
                break
        for idx in range(condition_range):
            if not os.path.exists(os.path.join(st_path, "C{}".format(c_list[idx]))):
                os.mkdir(os.path.join(st_path, "C{}".format(c_list[idx])))
            c_idx = c_list[idx]
            for idk in range(noise_range):
                _, n_idx = score_list[c_idx][idk][0]
                _, last_n_idx = score_list[c_idx][-1][0]
                up_score = score_list[c_idx][idk][1]
                down_score = score_list[c_idx][-1][1]

                up_pos = np.expand_dims(A[n_idx, c_idx,: , :], axis=0)
                down_pos = np.expand_dims(A[last_n_idx, c_idx,: , :], axis=0)
                save_name = "{}_SU{:.3f}_SD{:.3f}_C{}_NU{}_ND{}.mp4".format(idk, up_score, down_score, c_idx, n_idx, last_n_idx)
                save_path = os.path.join(st_path, "C{}".format(c_idx), save_name)
                BothSide(up_pos, down_pos, save_path)
    
    elif mode == 3:
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 10
        noise_range = 10
        if not os.path.exists(st_path):
            os.mkdir(st_path)
        c_list = []

        score_list = [((i, j), scores[j, i])  for j in range(noise_size) for i in range(condition_size)]
        score_list.sort(key=itemgetter(1), reverse=True)
        for idx in range(condition_size * noise_size):
            c_idx, n_idx = score_list[idx][0]
            if c_idx in c_list:
                continue
            else:
                c_list.append(c_idx)
            score = score_list[idx][1]
            pos = np.expand_dims(A[n_idx, c_idx,: , :], axis=0)
            save_name = "{}_S{}_C{}_N{}.mp4".format(idx, score, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            SingleSide(pos, save_path)
            if len(c_list) > condition_range:
                break

    elif mode == 4:
        scores1 = np.load(sc1_path)
        scores2 = np.load(sc2_path)
        A = np.load(rf_path)
        B = np.load(rr_path)
        condition_size = scores1.shape[1]
        noise_size = scores1.shape[0]
        condition_range = 10
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)


        score1_list = [ [ ((i, j), scores1[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=True)
        score2_list = [ [ ((i, j), scores2[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score2_list:
            each.sort(key=itemgetter(1), reverse=True)

        allsort_list = [((i, j), scores2[j, i])  for j in range(noise_size) for i in range(condition_size)]
        allsort_list.sort(key=itemgetter(1), reverse=True)

        c_list = []
        for idx in range(condition_size * noise_size):
            c_idx, B_n_idx = allsort_list[idx][0]
            _, A_n_idx = score1_list[c_idx][0][0]
            if c_idx in c_list:
                continue
            else:
                c_list.append(c_idx)
            sc1 = score1_list[c_idx][0][1]
            sc2 = score2_list[c_idx][0][1]
            up_pos = np.expand_dims(A[A_n_idx, c_idx,: , :], axis=0)
            down_pos = np.expand_dims(B[B_n_idx, c_idx, :, :], axis=0)

            save_name = "{}_SA{}_SB{}_C{}_AN{}_BN{}.mp4".format(idx, sc1, sc2, c_idx, A_n_idx, B_n_idx)
            save_path = os.path.join(st_path, save_name)
            BothSide(up_pos, down_pos, save_path)
            if len(c_list) > condition_range:
                break

    elif mode == 5:
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 10
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)
        for idx in range(condition_range):
            if not os.path.exists(os.path.join(st_path, "C{}".format(idx))):
                os.mkdir(os.path.join(st_path, "C{}".format(idx)))

            for idk in range(noise_range):
                score = scores[idk, idx]
                pos = np.expand_dims(A[idk, idx,: , :], axis=0)
                save_name = "{}_S{:.3f}_C{}_N{}.mp4".format(idk, score, c_idx, idk)
                save_path = os.path.join(st_path, "C{}".format(c_idx), save_name)
                SingleSide(up_pos, down_pos, save_path)

    elif mode == 6:
        scores = np.load(sc1_path)
        ##debug
        # input(scores.shape)

        A = np.load(rf_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 100
        noise_range = 10
        if not os.path.exists(st_path):
            os.mkdir(st_path)

        score1_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=True)

        for idc in range(len(select_list)):
            idk = select_list[idc]
            score = score1_list[idk][0][1]
            c_idx, n_idx = score1_list[idk][0][0]
            assert c_idx == idk
            ##debug
            # pos = np.expand_dims(A[n_idx, n_idx,: , :], axis=0)
            pos = np.expand_dims(A[n_idx, c_idx,: , :], axis=0)
            #debug
            # input(pos.shape)

            save_name = "{:02}_S{:.3f}_C{}_N{}.mp4".format(idc, score, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            SingleSide(data=pos, save_path=save_path, c_index=idc, n_index=None, n_frames=length_list[idk])
            ##debug


    elif mode == 7:
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        B = np.load(rr_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]


        if not os.path.exists(st_path):
            os.mkdir(st_path)

        score1_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=True)

        for idx in range(len(select_list)):
            idk = select_list[idx]
            score = score1_list[idk][0][1]
            c_idx, n_idx = score1_list[idk][0][0]
            assert c_idx == idk
            up_pos = np.expand_dims(A[n_idx, c_idx, :, :], axis=0)
            down_pos = np.expand_dims(B[c_idx, :, :], axis=0)
            save_name = "{:02}_S{:.3f}_C{}_N{}.mp4".format(idk, score, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            BothSideSpecial(data_left=up_pos, data_right=down_pos, save_path=save_path, c_index=idx)

    if mode == 8:
        #down is real ,up is fake if M == 1
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        B = np.load(rr_path)
        print("A shape ", A.shape)
        print("B shape ", B.shape)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 10
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)


        score_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score_list:
            each.sort(key=itemgetter(1), reverse=True)
        ##Denote Fake VS Real flipping
        mark = 0
        for idx in select_list:
            c_idx, n_idx = score_list[idx][0][0]
            score = score_list[idx][0][1]
            up_pos = np.expand_dims(A[n_idx, c_idx, :, :], axis=0)
            down_pos = np.expand_dims(B[c_idx, :, :], axis=0)
            save_name = "{}_sS{}_M{}_C{}_N{}.mp4".format(idx, score, mark, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            if mark == 0:
                BothSide(up_pos, down_pos, save_path)
                mark = 1
            else:
                BothSide(down_pos, up_pos, save_path)
                mark = 0

    if mode == 9:
        #down is real ,up is fake if M == 1
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        B = np.load(rr_path)
        print("A shape ", A.shape)
        print("B shape ", B.shape)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 100
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)


        score_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score_list:
            each.sort(key=itemgetter(1), reverse=True)
        ##Denote Fake VS Real flipping
        mark = 0
        for idx in range(condition_range):
            

            c_idx, n_idx = score_list[idx][-1][0]
            score = score_list[idx][-1][1]
            up_pos = np.expand_dims(A[n_idx, c_idx, :, :], axis=0)
            down_pos = np.expand_dims(B[c_idx, :, :], axis=0)
            save_name = "{}_M_{}_HS{}_C{}_N{}.mp4".format(idx, mark, score, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            if mark == 0:
                BothSide(up_pos, down_pos, save_path)
                mark = 1
            else:
                BothSide(down_pos, up_pos, save_path)
                mark = 0


    if mode == 10:
        #down is real ,up is fake if M == 1
        scores = np.load(sc1_path)
        print("Score shape", scores.shape)
        A = np.load(rf_path)
        print("RF shape ", A.shape)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 10
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)
        score_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        allsort_list = [((i, j), scores[j, i])  for j in range(noise_size) for i in range(condition_size)]
        for each in score_list:
            each.sort(key=itemgetter(1), reverse=True)

        mark = 0
        for idx in select_list:
            c_idx, n_idx = score_list[idx][0][0]
            low_c, low_n = score_list[c_idx][-1][0]
            high_c, high_n = score_list[c_idx][0][0]
            
            mark = 0
            up_pos = np.expand_dims(A[low_n, low_c, : , :], axis=0)
            down_pos = np.expand_dims(A[high_n, high_c, : , :], axis=0)
            save_name = "{}_M{}_L{}_H{}_HN_{}_LN_{}.mp4".format(c_idx, mark, low_c, high_c, high_n, low_n)
            save_path = os.path.join(st_path, save_name)
            if mark == 0:
                BothSide(up_pos, down_pos, save_path)
                mark = 1
            else:
                BothSide(down_pos, up_pos, save_path)
                mark = 0

    elif mode == 11:
        scores = np.load(sc1_path)
        B = np.load(rr_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 100
        noise_range = 10
        if not os.path.exists(st_path):
            os.mkdir(st_path)

        for idc in range(len(select_list)):
            idk = select_list[idc]
            ##debug
            # pos = np.expand_dims(A[n_idx, n_idx,: , :], axis=0)
            pos = np.expand_dims(B[idk,: , :], axis=0)
            save_name = "{}_S_C{}.mp4".format(idc, idk)
            save_path = os.path.join(st_path, save_name)
            SingleSide(pos, save_path)


    elif mode == 12:
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 100
        noise_range = 10
        if not os.path.exists(st_path):
            os.mkdir(st_path)

        score1_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=True)

        for idc in range(condition_range):
            score = score1_list[idc][0][1]
            c_idx, n_idx = score1_list[idc][0][0]
            # assert c_idx == idk
            ##debug
            # pos = np.expand_dims(A[n_idx, n_idx,: , :], axis=0)
            pos = np.expand_dims(A[n_idx, c_idx,: , :], axis=0)
            save_name = "{}_S{:.3f}_C{}_N{}.mp4".format(idc, score, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            SingleSide(pos, save_path)

    elif mode == 13:
        scores = np.load(sc1_path)
        A = np.load(rf_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 100
        noise_range = 10
        if not os.path.exists(st_path):
            os.mkdir(st_path)

        score1_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=True)

        for idc in range(condition_range):
            score = score1_list[idc][0][1]
            c_idx, n_idx = score1_list[idc][0][0]
            # assert c_idx == idk
            ##debug
            # pos = np.expand_dims(A[n_idx, n_idx,: , :], axis=0)
            pos = np.expand_dims(A[n_idx, c_idx,: , :], axis=0)
            save_name = "{}_S{:.3f}_C{}_N{}.mp4".format(idc, score, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            SingleSide(pos, save_path)

    if mode == 14:
        #down is real ,up is fake if M == 1
        scores = np.load(sc1_path)
        print("Score shape", scores.shape)
        A = np.load(rf_path)
        print("RF shape ", A.shape)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 10
        noise_range = 10

        if not os.path.exists(st_path):
            os.mkdir(st_path)
        
        allsort_list = [((i, j), scores[j, i])  for j in range(noise_size) for i in range(condition_size)]

        allsort_list.sort(key=itemgetter(1), reverse=True)

        mark = 0
        for idx in select_list:
            low_c, low_n = allsort_list[-1][0]
            high_c, high_n = allsort_list[idx][0]
            

            up_pos = np.expand_dims(A[low_n, low_c, : , :], axis=0)
            down_pos = np.expand_dims(A[high_n, high_c, : , :], axis=0)
            save_name = "{}_M{}_L{}_H{}_HN_{}_LN_{}.mp4".format(idx, mark, low_c, high_c, high_n, low_n)
            save_path = os.path.join(st_path, save_name)
            BothSide(up_pos, down_pos, save_path)

    #Mode 15 for CNN RNN Comparison
    elif mode == 15:
        scores1 = np.load(sc1_path)
        scores2 = np.load(sc2_path)
        A = np.load(rf_path)
        B = np.load(rr_path)
        condition_size = scores1.shape[1]
        noise_size = scores1.shape[0]


        if not os.path.exists(st_path):
            os.mkdir(st_path)


        score1_list = [ [ ((i, j), scores1[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=True)
        score2_list = [ [ ((i, j), scores2[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score2_list:
            each.sort(key=itemgetter(1), reverse=True)

        for idc in range(len(select_list)):
            idk = select_list[idc]
            c_idx_A, n_idx_A = score1_list[idk][0][0]
            c_idx_B, n_idx_B = score2_list[idk][0][0]
            assert c_idx_A == idk and c_idx_B == idk

            up_pos = np.expand_dims(A[n_idx_A, idk,: , :], axis=0)
            down_pos = np.expand_dims(B[n_idx_B, idk, :, :], axis=0)

            save_name = "{}_CNN_VS_RNN.mp4".format(idc)
            save_path = os.path.join(st_path, save_name)
            BothSideSpecial(data_left=up_pos, data_right=down_pos, save_path=save_path, c_index=idc)

    ##mode 16 for bad cases
    elif mode == 16:
        scores = np.load(sc1_path)
        ##debug
        # input(scores.shape)

        A = np.load(rf_path)
        condition_size = scores.shape[1]
        noise_size = scores.shape[0]
        condition_range = 100
        noise_range = 10
        if not os.path.exists(st_path):
            os.mkdir(st_path)

        score1_list = [ [ ((i, j), scores[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=False)

        for idc in range(len(select_list)):
            idk = select_list[idc]
            score = score1_list[idk][0][1]
            c_idx, n_idx = score1_list[idk][0][0]
            assert c_idx == idk
            ##debug
            # pos = np.expand_dims(A[n_idx, n_idx,: , :], axis=0)
            pos = np.expand_dims(A[n_idx, c_idx,: , :], axis=0)
            #debug
            # input(pos.shape)

            save_name = "{:02}_S{:.3f}_C{}_N{}.mp4".format(idc, score, c_idx, n_idx)
            save_path = os.path.join(st_path, save_name)
            SingleSide(data=pos, save_path=save_path, c_index=idc, n_index=None, n_frames=length_list[idk])

    ##wide open penalty
    elif mode == 17:
        scores1 = np.load(sc1_path)
        scores2 = np.load(sc2_path)
        A = np.load(rf_path)
        B = np.load(rr_path)
        condition_size = scores1.shape[1]
        noise_size = scores1.shape[0]


        if not os.path.exists(st_path):
            os.mkdir(st_path)


        score1_list = [ [ ((i, j), scores1[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score1_list:
            each.sort(key=itemgetter(1), reverse=True)
        score2_list = [ [ ((i, j), scores2[j, i])  for j in range(noise_size)] for i in range(condition_size)]
        for each in score2_list:
            each.sort(key=itemgetter(1), reverse=True)

        for idc in range(len(select_list)):
            idk = select_list[idc]
            c_idx_A, n_idx_A = score1_list[idk][0][0]
            c_idx_B, n_idx_B = score2_list[idk][0][0]
            assert c_idx_A == idk and c_idx_B == idk

            up_pos = np.expand_dims(A[n_idx_A, idk,: , :], axis=0)
            down_pos = np.expand_dims(B[n_idx_B, idk, :, :], axis=0)

            save_name = "{:02}_CNN_VS_RNN_Rest.mp4".format(idc)
            save_path = os.path.join(st_path, save_name)
            BothSideSpecial(data_left=up_pos, data_right=down_pos, save_path=save_path, c_index=idc)

# def update_both(i, player_circles_left, ball_circle_left, player_circles_right, ball_circle_right, annotations, data_left, data_right, factor):
#     for j, circle in enumerate(player_circles_left):
#         # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
#         circle.center = data_left[0, i , j * 2 +3] * factor, data_left[0, i , j * 2 +4] * factor
#         annotations[j].set_position(circle.center)
    
#     ball_circle_left.center = data_left[0, i , 0] * factor, data_left[0, i, 1] * factor
#     annotations[10].set_position(ball_circle_left.center)

#     offset = (100 - 6) * factor + 80
#     for j, circle in enumerate(player_circles_right):
#         # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
#         circle.center = data_right[0, i , j * 2 +3] * factor + offset, data_right[0, i , j * 2 +4] * factor
#         annotations[j + 11].set_position(circle.center)
    
#     ball_circle_right.center = data_right[0, i , 0] * factor + offset, data_right[0, i, 1] * factor
#     annotations[21].set_position(ball_circle_right.center)



#     return player_circles_left, ball_circle_left, player_circles_right, ball_circle_right



# def BothSide(data_left, data_right, save_path):

#     factor = 8
#     # one event
#     print(data_left.shape, data_right.shape)
#     go = np.concatenate((data_left, data_right), axis=2)
#     name = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'O', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'O']
#     color = ['black' for _ in range(22)]
#     ##offset = 100
#     ax = plt.axes(xlim=(0, 94 * factor * 2 + 80), ylim=(0,50 * factor))
#     print("length", len(name), len(color))
#     ax.axis('off')
#     fig = plt.gcf()
#     ax.grid(False)

#     player_circles_left = [plt.Circle((0,0), 15, color='r') for i in range(5)] + [plt.Circle((0,0), 15, color='b') for i in range(5)]
#     ball_circle_left = plt.Circle((0,0), 12, color='g')

#     player_circles_right = [plt.Circle((0,0), 15, color='r') for i in range(5)] + [plt.Circle((0,0), 15, color='b') for i in range(5)]
#     ball_circle_right = plt.Circle((0,0), 12, color='g')


#     for circle in player_circles_left:
#         ax.add_patch(circle)
#     ax.add_patch(ball_circle_left)

#     for circle in player_circles_right:
#         ax.add_patch(circle)
#     ax.add_patch(ball_circle_right)

#     annotations = [ax.annotate(name[i], xy=[0, 0], color=color[i],
#                     horizontalalignment='center',
#                     verticalalignment='center', fontweight='bold',fontsize=6)
#                     for i in range(22)]

#     anim = animation.FuncAnimation(fig, update_both, fargs=(player_circles_left, ball_circle_left, player_circles_right, ball_circle_right, annotations, data_left, data_right, factor), frames=100, interval=100)
    
#     court = plt.imread("bothfull.png")
#     plt.imshow(court, zorder=0, extent=[0, 94 * factor * 2 + 80, 50 * factor, 0])
#     #if save:
    
#     anim.save(save_path, dpi=200,fps=4, writer='ffmpeg')
#     plt.close(fig)
#     print("Saved ", save_path)