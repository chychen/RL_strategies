from __future__ import print_function
import argparse
from operator import itemgetter
import os
from os import listdir
from os.path import join
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc
import math
from utilsA import Visualization
import random
import numpy as np

parser = argparse.ArgumentParser(description='NBA play visualization')
parser.add_argument('--m',type=int, default=1, help='Mode: 1: Real VS Fake\
                                                           2: Fake_1st VS Fake_last\
                                                           3: Sorted Fake\
                                                           4: Fake Comparison\
                                                           5: Variance Comparison')
parser.add_argument('--rf', default='', help='data path to RF(Real_A-Fake_B pair) data')
parser.add_argument('--rr', default='', help='data path to RR(Real_A-Real_B pair) data')
parser.add_argument('--sc1', help='path to rf1 scores data', default='results_critic_scores')
parser.add_argument('--sc2', help='path to rf2 scores data', default='')
parser.add_argument('--st', default='' ,help='store path ')


options = parser.parse_args()
print("Mode: {}".format(options.m))
print('RF path: {}'.format(options.rf))
print("RR path: {}".format(options.rr))
print("SC1 path: {}".format(options.sc1))
print("SC2 path: {}".format(options.sc2))
print("ST path: {}".format(options.st))

mode = options.m
rf_path = options.rf
rr_path = options.rr
sc1_path = options.sc1
sc2_path = options.sc2
st_path = options.st
#""vf2_464k_WP0H50ex_valid/results_A_fake_B.npy"/results_A_fake_B.npy"


# shape (100, 1152, 100, 23)
rf_path = "vf2_464k_WP0H50ex_valid/mode_1/results_A_fake_B.npy"
rr_path = "New_RNN/mode_1/results_A_fake_B.npy"
sc1_path = "vf2_464k_WP0H50ex_valid/mode_1/results_critic_scores.npy"
sc2_path = "New_RNN/mode_1/results_critic_scores.npy"
# sc1_path = sc2_path
st_path = 'Output_Model6_vf2_464k_WP0H50ex_valid_New_RNN_Rest_OhNo_Ex/'

## selected index
choice = []






Visualization(mode=17, rf_path=rf_path, rr_path=rr_path, \
              sc1_path=sc1_path, sc2_path=sc2_path, st_path=st_path1, select_list=choice, length_list=None, name_list=None)

# Visualization(mode=8, rf_path=rf_path, rr_path=rr_path, \
#               sc1_path=sc1_path, sc2_path=sc2_path, st_path=st_path2, select_list=random_fake[3:])


# Visualization(mode=10, rf_path=rf_path, rr_path=rr_path, \
#               sc1_path=sc1_path, sc2_path=sc2_path, st_path=st_path3, select_list=random_fake)


# st_path4 = st_path + "Mode11"
# if not os.path.exists(st_path4):
#     os.mkdir(st_path4)

# Visualization(mode=11, rf_path=rf_path, rr_path=rr_path, \
#               sc1_path=sc1_path, sc2_path=sc2_path, st_path=st_path4, select_list=random_real)

# Visualization(mode=9, rf_path=rf_path, rr_path=rr_path, \
#               sc1_path=sc1_path, sc2_path=sc2_path, st_path=st_path, select_list=random_real)