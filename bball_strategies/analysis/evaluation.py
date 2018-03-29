"""
EvaluationMatrix:
- show_mean_distance():
    Mean/Stddev of distance between offense (wi/wo ball) and defense
- show_overlap_freq():
    Overlap frequency (judged by threshold = radius 1 ft)
- plot_histogram_vel_acc():
    Histogram of DEFENSE's velocity and acceleration. (mean,stddev)
- show_freq_heatmap():
    Vis Heat map (frequency) of positions
- show_best_match():
    Best match between real and defense’s position difference.(mean,stddev)
- show_freq_of_valid_defense():
    Compare to formula (defense sync with offense movement)
- plot_linechart_distance_by_frames():
    Vis dot distance of each offense to closest defense frame by frame
    (with indicators: inside 3pt line, ball handler, paint area)
- show_freq_heatmap():
    Vis generated defense player position by heatmap of
- plot_histogram_distance_by_frames():
    plot histogram of distance frame by frame, withball and without ball
- vis_and_analysis_by_episode():
    given episode index, draw out the play and show all matrix result in same folder

TODO:
- collision
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import colorsys
import itertools
import vis_game
import shutil


class EvaluationMatrix(object):
    """ EvaluationMatrix
    """

    def __init__(self, file_name='default', length=None, FORMULA_RADIUS=None, FPS=5, **kwargs):
        """ all data in keargs must have the shape=(num_episode, length, 23)

        Args
        ----
        kwargs : datasets, all items' values should have the same shape as real_data.shape
            key = dataset name
            value = dataset value

        length : list, shape=(num_episode,)
            true lenght of each episode, if None = true length are equal to data.shape[1]

        Raise
        -----
        ValueError :
            while init, kwargs must contains the 'real_data', or raise error.
            kwargs['real_data'], float, shape=(num_episode, length, 23), 23 = ball(3) + off(10) + def(10)
        """
        if kwargs['real_data'] is None:
            raise ValueError('You should pass \'real_data\' in arguments')

        if length is None:
            length = kwargs['real_data'].shape[0]

        self.RIGHT_BASKET = [94 - 5.25, 25]
        self.WINGSPAN_RADIUS = 3.5 + 0.5  # to judge who handle the ball
        self.LEN_3PT_BASKET = 23.75 + 5
        self.FPS = FPS

        self._file_name = file_name
        self._all_data_dict = kwargs
        self._length = length
        self._num_episodes = self._length.shape[0]

        # mkdir
        if not os.path.exists(self._file_name):
            os.makedirs(self._file_name)
        # report.txt
        report_file_path = os.path.join(self._file_name, "report.txt")
        if os.path.exists(report_file_path):
            open(report_file_path, "w").close()  # empty the file
        self._report_file = open(report_file_path, "a")

        if FORMULA_RADIUS is not None:
            self.create_formula_1_defense(RADIUS=FORMULA_RADIUS)
            self.create_formula_2_defense(RADIUS=FORMULA_RADIUS)

    def show_overlap_freq(self, OVERLAP_RADIUS=1.0, interp_flag=True):
        """ Overlap frequency (judged by threshold = OVERLAP_RADIUS)
        """
        show_msg = '\n### show_overlap_freq ###\n'
        for key, data in self._all_data_dict.items():
            interp_frame = 10
            if interp_flag:
                data = self.__interp_frame(data, interp_frame=interp_frame)
            offense = np.reshape(data[:, :, 3:13], [
                data.shape[0], data.shape[1], 5, 2])
            defense = np.reshape(data[:, :, 13:23], [
                data.shape[0], data.shape[1], 5, 2])
            total_frames = data.shape[0]*data.shape[1]

            counter = 0
            for off_idx in range(5):
                temp_len = self.__get_length(
                    defense, offense[:, :, off_idx:off_idx+1])
                # clean up unused length
                for i in range(data.shape[0]):
                    if interp_flag:
                        temp_len[i, (self._length[i] - 1) *
                                 interp_frame:] = np.inf
                    else:
                        temp_len[i, self._length[i]:] = np.inf
                counter += np.count_nonzero(temp_len <= OVERLAP_RADIUS)
            # show
            show_msg += '\'{0}\' dataset\n'.format(
                key) + '-- frequency={0:4.2f}\n'.format(counter/total_frames)
        print(show_msg)
        self._report_file.write(show_msg)

    def show_mean_distance(self, mode='THETA'):
        """ Mean/Stddev of distance between offense (wi/wo ball) and defense
        """
        show_msg = '\n### show_mean_distance ###\n'
        for key, data in self._all_data_dict.items():
            dist = self.__evalute_distance(data, mode=mode)
            ball = np.reshape(data[:, :, 0:2], [
                data.shape[0], data.shape[1], 1, 2])
            offense = np.reshape(data[:, :, 3:13], [
                data.shape[0], data.shape[1], 5, 2])
            indices = self.__get_length(offense, ball) < self.WINGSPAN_RADIUS
            # clean up unused length
            for i in range(data.shape[0]):
                indices[i, self._length[i]:] = False
            # with ball
            wi_mean = np.mean(dist[indices])
            wi_std = np.std(dist[indices])
            # without ball
            indices = np.logical_not(indices)
            # clean up unused length
            for i in range(data.shape[0]):
                indices[i, self._length[i]:] = False
            wo_mean = np.mean(dist[indices])
            wo_std = np.std(dist[indices])
            # show
            show_msg += '\'{0}\' dataset\n'.format(key) + '-- wi ball:\tmean={0:4.2f},\tstddev={1:4.2f}\n'.format(
                wi_mean, wi_std) + '-- wo ball:\tmean={0:4.2f},\tstddev={1:4.2f}\n'.format(wo_mean, wo_std)
        print(show_msg)
        self._report_file.write(show_msg)

    def __interp_frame(self, data, interp_frame=10):

        res = []
        for epi in data:
            x = np.arange(0, len(epi) - 1, 1.0 / interp_frame)
            xp = np.arange(0, len(epi), 1.0)
            tmp = []
            for idx in range(23):
                fp = epi[:, idx]
                y = np.interp(x, xp, fp)
                tmp.append(y)
            res.append(tmp)

        res = np.asarray(res)
        res = np.einsum('ijk->ikj', res)

        return res

    def __get_length(self, a, b, axis=-1):
        """ get distance between a and b by axis
        """
        vec = a-b
        return np.sqrt(np.sum(vec*vec, axis=axis))

    def __get_visual_aid(self, data):
        """ return markers

        Return
        ------
        result = {}
        result['handler_idx'] : shape=[num_episode, pad_length, 5]
        result['if_inside_3pt'] : shape=[num_episode, pad_length, 5]
        result['if_inside_paint'] : shape=[num_episode, pad_length, 5]
        """
        ball = np.reshape(data[:, :, 0:2], [
            data.shape[0], data.shape[1], 1, 2])
        offense = np.reshape(data[:, :, 3:13], [
            data.shape[0], data.shape[1], 5, 2])
        pad_next = np.pad(offense[:, 1:], [(0, 0), (0, 1),
                                           (0, 0), (0, 0)], mode='constant', constant_values=1)
        offense_speed = self.__get_length(offense, pad_next) * self.FPS
        offense_speed[:, -1] = None

        handler_idx = np.empty(
            shape=[data.shape[0], data.shape[1], 5], dtype=object)
        if_inside_3pt = np.empty(
            shape=[data.shape[0], data.shape[1], 5], dtype=object)
        if_inside_paint = np.empty(
            shape=[data.shape[0], data.shape[1], 5], dtype=object)

        for off_idx in range(5):
            offender = offense[:, :, off_idx:off_idx+1, :]
            # mark frames when driible
            indices = np.where(self.__get_length(
                offender[:, :, 0], ball[:, :, 0]) < self.WINGSPAN_RADIUS)
            handler_idx[indices[0], indices[1], off_idx] = -2  # dummy constant
            # check position whether inside the 3pt line
            indices = np.where(self.__get_length(
                offender[:, :, 0], self.RIGHT_BASKET) < self.LEN_3PT_BASKET)
            if_inside_3pt[indices[0], indices[1],
                          off_idx] = -4  # dummy constant
            # check position whether inside the paint area
            judge_paint = np.logical_and(
                offender[:, :, 0, 0] < 94, offender[:, :, 0, 0] >= 75)
            judge_paint = np.logical_and(
                judge_paint, offender[:, :, 0, 1] < 33)
            judge_paint = np.logical_and(
                judge_paint, offender[:, :, 0, 1] >= 17)
            indices = np.where(judge_paint)
            if_inside_paint[indices[0], indices[1],
                            off_idx] = -6  # dummy constant

        # clean up unused length
        for i in range(data.shape[0]):
            handler_idx[i, self._length[i]:, :] = None
            if_inside_3pt[i, self._length[i]:, :] = None
            if_inside_paint[i, self._length[i]:, :] = None
        result = {}
        result['handler_idx'] = handler_idx
        result['if_inside_3pt'] = if_inside_3pt
        result['if_inside_paint'] = if_inside_paint
        # result['offense_speed'] = offense_speed

        return result

    def __evalute_distance(self, data, mode='THETA'):
        """ evaluate the distance to the closest defender for each offensive player on each frames
        and mark the offensive player who has the ball according to self.WINGSPAN_RADIUS

        Args
        ----
        data : float, shape=[num_episodes, padded_length, 23]
            the positions includes ball, offense, and defense

        Returns
        -------
        dist : float, shape=[num_episodes, padded_length, 5]
            the distance to the closest defender for each offensive player
        """
        ball = np.reshape(data[:, :, 0:2], [
            data.shape[0], data.shape[1], 1, 2])
        offense = np.reshape(data[:, :, 3:13], [
            data.shape[0], data.shape[1], 5, 2])
        pad_next = np.pad(offense[:, 1:], [(0, 0), (0, 1),
                                           (0, 0), (0, 0)], mode='constant', constant_values=1)
        offense_speed = self.__get_length(offense, pad_next) * self.FPS
        offense_speed[:, -1] = None
        defense = np.reshape(data[:, :, 13:23], [
            data.shape[0], data.shape[1], 5, 2])

        dist = np.zeros(shape=[data.shape[0], data.shape[1], 5])

        for off_idx in range(5):
            offender = offense[:, :, off_idx:off_idx+1, :]

            if mode == 'DISTANCE':
                # the distance to the closest defender
                off2defs = defense - offender
                off2basket = self.RIGHT_BASKET - offender
                dotvalue = off2defs[:, :, :, 0] * off2basket[:, :, :,
                                                             0] + off2defs[:, :, :, 1] * off2basket[:, :, :, 1]
                off2defs_len = self.__get_length(offender, defense)
                # find best defense according to defense_scores
                defense_scores = np.array(off2defs_len)
                defense_scores[dotvalue <= 0] = np.inf
                best_defense_idx = np.argmin(defense_scores, axis=-1)
                for i in range(dist.shape[0]):
                    for j in range(dist.shape[1]):
                        dist[i, j, off_idx] = off2defs_len[i,
                                                           j, best_defense_idx[i, j]]
            elif mode == 'THETA':
                # the lowest score = (theta+1) * (distance+1)
                off2defs = defense - offender
                off2basket = self.RIGHT_BASKET - offender
                dotvalue = off2defs[:, :, :, 0] * off2basket[:, :, :,
                                                             0] + off2defs[:, :, :, 1] * off2basket[:, :, :, 1]
                off2defs_len = self.__get_length(offender, defense)
                # find best defense according to defense_scores
                defense_scores = (np.arccos(dotvalue/(self.__get_length(defense, offender)
                                                      * self.__get_length(self.RIGHT_BASKET, offender)+1e-8))+1.0)*(off2defs_len+1.0)
                defense_scores[dotvalue <= 0] = np.inf
                best_defense_idx = np.argmin(defense_scores, axis=-1)
                for i in range(dist.shape[0]):
                    for j in range(dist.shape[1]):
                        dist[i, j, off_idx] = off2defs_len[i,
                                                           j, best_defense_idx[i, j]]
            elif mode == 'THETA_ADD_SCORE':
                # the lowest score = (theta*10) + (distance+1)
                off2defs = defense - offender
                off2basket = self.RIGHT_BASKET - offender
                dotvalue = off2defs[:, :, :, 0] * off2basket[:, :, :,
                                                             0] + off2defs[:, :, :, 1] * off2basket[:, :, :, 1]
                off2defs_len = self.__get_length(offender, defense)
                # find best defense according to defense_scores
                defense_scores = (np.arccos(dotvalue/(self.__get_length(defense, offender)
                                                      * self.__get_length(self.RIGHT_BASKET, offender)+1e-4))*10.0)+off2defs_len
                # defense_scores[dotvalue <= 0] = np.inf  # avoid nan and no need
                best_defense_idx = np.argmin(defense_scores, axis=-1)
                for i in range(dist.shape[0]):
                    for j in range(dist.shape[1]):
                        dist[i, j, off_idx] = defense_scores[i,
                                                             j, best_defense_idx[i, j]]
            elif mode == 'THETA_MUL_SCORE':
                # the lowest score = (theta+1) * (distance+1)
                off2defs = defense - offender
                off2basket = self.RIGHT_BASKET - offender
                dotvalue = off2defs[:, :, :, 0] * off2basket[:, :, :,
                                                             0] + off2defs[:, :, :, 1] * off2basket[:, :, :, 1]
                off2defs_len = self.__get_length(offender, defense)
                # find best defense according to defense_scores
                defense_scores = (np.arccos(dotvalue/(self.__get_length(defense, offender)
                                                      * self.__get_length(self.RIGHT_BASKET, offender)+1e-4))+1.0)*(off2defs_len+1.0)
                # defense_scores[dotvalue <= 0] = np.inf  # avoid nan and no need
                best_defense_idx = np.argmin(defense_scores, axis=-1)
                for i in range(dist.shape[0]):
                    for j in range(dist.shape[1]):
                        dist[i, j, off_idx] = defense_scores[i,
                                                             j, best_defense_idx[i, j]]

        # clean up unused length
        for i in range(data.shape[0]):
            dist[i, self._length[i]:] = 0.0

        return dist

    def plot_linechart_distance_by_frames(self, mode='THETA'):
        """ Vis dot distance of each offense to closest defense frame by frame
        (with indicators: inside 3pt line, ball handler, paint area)

        Args
        ----
        handler_idx : int, shape=[num_episodes, padded_length, 5]
            one hot vector represent the ball handler idx for each frame
        if_inside_3pt :
        if_inside_paint :
        """
        print('\n### plot_linechart_distance_by_frames ###\n')
        # caculate the matrix
        all_dist_dict = {}
        for key, data in self._all_data_dict.items():
            all_dist_dict[key] = self.__evalute_distance(data, mode=mode)
        all_marker_dict = self.__get_visual_aid(
            self._all_data_dict['real_data'])
        # mkdir
        save_path = os.path.join(self._file_name, 'linechart_'+mode)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # color
        HSV_tuples = [((x % 10)/float(10), 0.8, 0.8)
                      for x in range(0, 3*10, 3)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
        # vis
        for epi_idx in range(self._num_episodes):
            color_count = 0
            all_trace = []
            epi_len = self._length[epi_idx]
            for i, [key, data] in enumerate(all_dist_dict.items()):
                for off_idx in range(5):
                    # real
                    trace = go.Scatter(
                        x=np.arange(epi_len)/6.25,
                        y=data[epi_idx, :epi_len, off_idx],
                        name=key+'_'+str(off_idx+1),
                        xaxis='x',
                        yaxis='y'+str(off_idx+1),
                        line=dict(
                            color=('rgb'+str(RGB_tuples[color_count]))
                        )
                    )
                    all_trace.append(trace)
                color_count += 1
            for i, [key, data] in enumerate(all_marker_dict.items()):
                for off_idx in range(5):
                    # markers
                    trace = go.Scatter(
                        x=np.arange(epi_len)/6.25,
                        y=data[epi_idx, :epi_len, off_idx],
                        name=key+'_'+str(off_idx+1),
                        xaxis='x',
                        yaxis='y'+str(off_idx+1),
                        line=dict(
                            color=('rgb'+str(RGB_tuples[color_count])),
                            width=3)
                    )
                    all_trace.append(trace)
                color_count += 1
            layout = go.Layout(
                xaxis=dict(title='time (sec)'),
                yaxis1=dict(
                    title='player_1\'s distance (feet)',
                    domain=[0.0, 0.15]
                ),
                yaxis2=dict(
                    title='player_2\'s distance (feet)',
                    domain=[0.2, 0.35]
                ),
                yaxis3=dict(
                    title='player_3\'s distance (feet)',
                    domain=[0.4, 0.55]
                ),
                yaxis4=dict(
                    title='player_4\'s distance (feet)',
                    domain=[0.6, 0.75]
                ),
                yaxis5=dict(
                    title='player_5\'s distance (feet)',
                    domain=[0.8, 0.95]
                )
            )
            fig = go.Figure(data=all_trace, layout=layout)
            py.plot(fig, filename=os.path.join(
                save_path, 'epi_{}.html'.format(epi_idx)), auto_open=False)

    def plot_histogram_vel_acc(self):
        """ Histogram of DEFENSE's speed and acceleration. (mean,stddev)
        """
        msg = '\n### plot_histogram_vel_acc ###\n'
        # mkdir
        save_path = os.path.join(self._file_name, 'histogram')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # speed
        all_trace_speed = []
        msg += 'Speed\n'
        for key, data in self._all_data_dict.items():
            if key == 'real_data':  # vis offense tooooooo
                target = np.reshape(data[:, :, 3:13], [
                    data.shape[0], data.shape[1], 5, 2])
                speed = self.__get_length(
                    target[:, 1:], target[:, :-1]) * self.FPS
                # clean up unused length
                valid_speed = []
                for i in range(data.shape[0]):
                    valid_speed.append(
                        speed[i, :self._length[i]-1].reshape([-1, ]))
                valid_speed = np.concatenate(valid_speed, axis=0)

                bin_size = 0.5
                max_v = np.amax(valid_speed)
                min_v = np.amin(valid_speed)
                num_bins = int((max_v-min_v)//bin_size)+1
                counter = np.zeros(shape=[num_bins, ])
                for v in valid_speed:
                    counter[int((v-min_v)//bin_size)] += 1

                trace_speed = go.Scatter(
                    name=key+'_offense',
                    x=[i*bin_size for i in range(num_bins)],
                    y=counter
                )

                # trace_speed = go.Histogram(
                #     name=key+'_offense',
                #     x=valid_speed,
                #     autobinx=False,
                #     xbins=dict(
                #         start=0.0,
                #         end=1000.0,
                #         size=0.5
                #     ),
                #     opacity=0.5
                # )
                all_trace_speed.append(trace_speed)
                msg += '{0} dataset:\tmean={1:4.2f},\tstddev={2:4.2f}\n'.format(
                    key+'_offense', np.mean(valid_speed), np.std(valid_speed))

            target = np.reshape(data[:, :, 13:23], [
                data.shape[0], data.shape[1], 5, 2])
            speed = self.__get_length(
                target[:, 1:], target[:, :-1]) * self.FPS
            # clean up unused length
            valid_speed = []
            for i in range(data.shape[0]):
                valid_speed.append(
                    speed[i, :self._length[i]-1].reshape([-1, ]))
            valid_speed = np.concatenate(valid_speed, axis=0)

            bin_size = 0.5
            max_v = np.amax(valid_speed)
            min_v = np.amin(valid_speed)
            num_bins = int((max_v-min_v)//bin_size)+1
            counter = np.zeros(shape=[num_bins, ])
            for v in valid_speed:
                counter[int((v-min_v)//bin_size)] += 1

            trace_speed = go.Scatter(
                name=key,
                x=[i*bin_size for i in range(num_bins)],
                y=counter
            )

            # trace_speed = go.Histogram(
            #     name=key,
            #     x=valid_speed,
            #     autobinx=False,
            #     xbins=dict(
            #         start=0.0,
            #         end=1000.0,
            #         size=0.5
            #     ),
            #     opacity=0.5
            # )
            all_trace_speed.append(trace_speed)
            msg += '{0} dataset:\tmean={1:4.2f},\tstddev={2:4.2f}\n'.format(
                key, np.mean(valid_speed), np.std(valid_speed))
        layout_speed = go.Layout(
            title='Speed',
            barmode='overlay',
            xaxis=dict(title='speed (feet/sec)'),
            yaxis=dict(title='counts')
        )
        fig_speed = go.Figure(data=all_trace_speed, layout=layout_speed)
        py.plot(fig_speed, filename=os.path.join(
            save_path, 'speed_histogram.html'), auto_open=False)

        # acc
        msg += 'Acceleration\n'
        all_trace_acc = []
        for key, data in self._all_data_dict.items():
            if key == 'real_data':  # vis offense tooooooo
                target = np.reshape(data[:, :, 3:13], [
                    data.shape[0], data.shape[1], 5, 2])
                speed = self.__get_length(
                    target[:, 1:], target[:, :-1]) * self.FPS
                acc = self.__get_length(speed[:, 1:], speed[:, :-1])
                # clean up unused length
                valid_acc = []
                for i in range(data.shape[0]):
                    valid_acc.append(
                        acc[i, :self._length[i]-2].reshape([-1, ]))
                valid_acc = np.concatenate(valid_acc, axis=0)

                bin_size = 0.5
                max_v = np.amax(valid_acc)
                min_v = np.amin(valid_acc)
                num_bins = int((max_v-min_v)//bin_size)+1
                counter = np.zeros(shape=[num_bins, ])
                for v in valid_acc:
                    counter[int((v-min_v)//bin_size)] += 1

                trace_acc = go.Scatter(
                    name=key+'_offense',
                    x=[i*bin_size for i in range(num_bins)],
                    y=counter
                )

                # trace_acc = go.Histogram(
                #     name=key+'_offense',
                #     x=valid_acc,
                #     autobinx=False,
                #     xbins=dict(
                #         start=0.0,
                #         end=1000.0,
                #         size=0.5
                #     ),
                #     opacity=0.5
                # )
                all_trace_acc.append(trace_acc)
                msg += '{0} dataset:\tmean={1:4.2f},\tstddev={2:4.2f}\n'.format(
                    key+'_offense', np.mean(valid_acc), np.std(valid_acc))

            target = np.reshape(data[:, :, 13:23], [
                data.shape[0], data.shape[1], 5, 2])
            speed = self.__get_length(
                target[:, 1:], target[:, :-1]) * self.FPS
            acc = self.__get_length(speed[:, 1:], speed[:, :-1])
            # clean up unused length
            valid_acc = []
            for i in range(data.shape[0]):
                valid_acc.append(acc[i, :self._length[i]-2].reshape([-1, ]))

            valid_acc = np.concatenate(valid_acc, axis=0)
            bin_size = 0.5
            max_v = np.amax(valid_acc)
            min_v = np.amin(valid_acc)
            num_bins = int((max_v-min_v)//bin_size)+1
            counter = np.zeros(shape=[num_bins, ])
            for v in valid_acc:
                counter[int((v-min_v)//bin_size)] += 1

            trace_acc = go.Scatter(
                name=key,
                x=[i*bin_size for i in range(num_bins)],
                y=counter
            )

            # trace_acc = go.Histogram(
            #     name=key,
            #     x=valid_acc,
            #     autobinx=False,
            #     xbins=dict(
            #         start=0.0,
            #         end=1000.0,
            #         size=0.5
            #     ),
            #     opacity=0.5
            # )
            all_trace_acc.append(trace_acc)
            msg += '{0} dataset:\tmean={1:4.2f},\tstddev={2:4.2f}\n'.format(
                key, np.mean(valid_acc), np.std(valid_acc))
        layout_acc = go.Layout(
            title='Acceleration',
            barmode='overlay',
            xaxis=dict(title='acceleration (feet/sec^2)'),
            yaxis=dict(title='counts')
        )
        fig_acc = go.Figure(data=all_trace_acc, layout=layout_acc)
        py.plot(fig_acc, filename=os.path.join(
            save_path, 'acc_histogram.html'), auto_open=False)
        print(msg)
        self._report_file.write(msg)

    def show_best_match(self):
        """ Best match between real and defense’s position difference.(mean,stddev)
        """
        show_msg = '\n### show_best_match ###\n'
        real_data = self._all_data_dict['real_data']
        real_defense = np.reshape(real_data[:, :, 13:23], [
            real_data.shape[0], real_data.shape[1], 5, 2])
        for key, data in self._all_data_dict.items():
            if key == 'real_data':
                continue
            fake_defense = np.reshape(data[:, :, 13:23], [
                data.shape[0], data.shape[1], 5, 2])
            # greedy find all combinations
            greedy_table = np.empty(
                shape=[real_data.shape[0], real_data.shape[1], 5, 5])
            for real_idx in range(5):
                for fake_idx in range(5):
                    greedy_table[:, :, real_idx, fake_idx] = self.__get_length(
                        real_defense[:, :, real_idx], fake_defense[:, :, fake_idx])
            # Permutation = 5! = 120
            permu_list = np.empty(
                shape=[real_data.shape[0], real_data.shape[1], 5*4*3*2*1])
            permu_idx_list = list(itertools.permutations(range(5)))
            # find best match
            for i, combination in enumerate(permu_idx_list):
                temp_sum = 0.0
                for j, idx in enumerate(combination):
                    temp_sum += greedy_table[:, :, j, idx]
                permu_list[:, :, i] = temp_sum / 5.0  # mean
            permu_min = np.amin(permu_list, axis=-1)
            # clean up unused length
            valid_match = []
            for i in range(data.shape[0]):
                valid_match.append(
                    permu_min[i, :self._length[i]].reshape([-1]))
            valid_match = np.concatenate(valid_match, axis=0)
            mean = np.mean(valid_match)
            stddev = np.std(valid_match)
            # show
            show_msg += '\'{0}\' dataset compared to \'real\' dataset\n'.format(
                key) + '-- mean={0:4.2f},\tstddev={1:4.2f}\n'.format(mean, stddev)
        print(show_msg)
        self._report_file.write(show_msg)

    def plot_show_best_match(self, save_path):
        """ Best match between real and defense’s position difference.(mean,stddev)
        """
        show_msg = '\n### plot_show_best_match ###\n'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        real_data = self._all_data_dict['real_data']
        real_defense = np.reshape(real_data[:, :, 13:23], [
            real_data.shape[0], real_data.shape[1], 5, 2])
        for key, data in self._all_data_dict.items():
            if key == 'real_data':
                continue
            fake_defense = np.reshape(data[:, :, 13:23], [
                data.shape[0], data.shape[1], 5, 2])
            # greedy find all combinations
            greedy_table = np.empty(
                shape=[real_data.shape[0], real_data.shape[1], 5, 5])
            for real_idx in range(5):
                for fake_idx in range(5):
                    greedy_table[:, :, real_idx, fake_idx] = self.__get_length(
                        real_defense[:, :, real_idx], fake_defense[:, :, fake_idx])
            # Permutation = 5! = 120
            permu_list = np.empty(
                shape=[real_data.shape[0], real_data.shape[1], 5*4*3*2*1])
            permu_idx_list = list(itertools.permutations(range(5)))
            # find best match
            for i, combination in enumerate(permu_idx_list):
                temp_sum = 0.0
                for j, idx in enumerate(combination):
                    temp_sum += greedy_table[:, :, j, idx]
                permu_list[:, :, i] = temp_sum / 5.0  # mean
            permu_min = np.amin(permu_list, axis=-1)
            permu_argmin = np.argmin(permu_list, axis=-1)
            permu_idx_all = np.zeros(shape=[
                                     real_data.shape[0], real_data.shape[1], 5*4*3*2*1, 5], dtype=int) + np.array(permu_idx_list, dtype=int)
            best_match_pairs = np.empty(
                shape=[real_data.shape[0], real_data.shape[1], 5], dtype=int)
            for i, v in enumerate(permu_argmin[0]):
                best_match_pairs[0, i] = permu_idx_all[0, i, v]
            # plot best match as video
            vis_game.plot_compare_data(real_data[0], data[0], self._length[0], permu_min[0], best_match_pairs[0], file_path=os.path.join(
                save_path, key+'_compare_best_match.mp4'), if_save=True, fps=self.FPS, vis_ball_height=False, vis_annotation=False)
            # clean up unused length
            valid_match = []
            for i in range(data.shape[0]):
                valid_match.append(
                    permu_min[i, :self._length[i]].reshape([-1]))
            valid_match = np.concatenate(valid_match, axis=0)
            mean = np.mean(valid_match)
            stddev = np.std(valid_match)
            # show
            show_msg += '\'{0}\' dataset compared to \'real\' dataset\n'.format(
                key) + '-- mean={0:4.2f},\tstddev={1:4.2f}\n'.format(mean, stddev)
        print(show_msg)
        self._report_file.write(show_msg)

    def create_formula_1_defense(self, RADIUS):
        """ formula (defense sync with offense movement)
        """
        if 'formula_1_data' in self._all_data_dict:
            return
        else:
            print('\n### create_formula_1_defense ###\n')
            real_data = self._all_data_dict['real_data']
            real_ball = real_data[:, :, 0:3]
            real_offense = np.reshape(real_data[:, :, 3:13], [
                real_data.shape[0], real_data.shape[1], 5, 2])
            # real_defense = np.reshape(real_data[:, :, 13:23], [
            #     real_data.shape[0], real_data.shape[1], 5, 2])
            first_vec = self.RIGHT_BASKET - real_offense[:, 0:1]
            formula_defense = real_offense + first_vec*RADIUS / \
                (self.__get_length(self.RIGHT_BASKET,
                                   real_offense[:, 0:1])[:, :, :, None]+1e-8)
            formula_data = np.concatenate([real_ball, real_offense.reshape(
                [real_data.shape[0], real_data.shape[1], 10]), formula_defense.reshape([real_data.shape[0], real_data.shape[1], 10])], axis=-1)
            # clean up unused length
            for i in range(formula_data.shape[0]):
                formula_data[i, self._length[i]:] = 0.0
            # back to dataset format
            self._all_data_dict['formula_1_data'] = formula_data

    def create_formula_2_defense(self, RADIUS):
        """ formula (defense sync with offense movement)
        """
        if 'formula_2_data' in self._all_data_dict:
            return
        else:
            print('\n### create_formula_2_defense ###\n')
            real_data = self._all_data_dict['real_data']
            real_ball = real_data[:, :, 0:3]
            real_offense = np.reshape(real_data[:, :, 3:13], [
                real_data.shape[0], real_data.shape[1], 5, 2])
            # real_defense = np.reshape(real_data[:, :, 13:23], [
            #     real_data.shape[0], real_data.shape[1], 5, 2])
            off2basket = self.RIGHT_BASKET - real_offense
            formula_defense = real_offense + off2basket*RADIUS / \
                (self.__get_length(self.RIGHT_BASKET,
                                   real_offense)[:, :, :, None]+1e-8)
            formula_data = np.concatenate([real_ball, real_offense.reshape(
                [real_data.shape[0], real_data.shape[1], 10]), formula_defense.reshape([real_data.shape[0], real_data.shape[1], 10])], axis=-1)
            # clean up unused length
            for i in range(formula_data.shape[0]):
                formula_data[i, self._length[i]:] = 0.0
            # back to dataset format
            self._all_data_dict['formula_2_data'] = formula_data

    def show_freq_of_valid_defense(self, RADIUS=10.0, THETA=10.0):
        """ validate defense by RADIUS and +-THETA
        """
        show_msg = '\n### show_freq_cmp_to_formula ###\n'
        for key, data in self._all_data_dict.items():
            ball = np.reshape(data[:, :, 0:2], [
                data.shape[0], data.shape[1], 1, 2])
            offense = np.reshape(data[:, :, 3:13], [
                data.shape[0], data.shape[1], 5, 2])
            defense = np.reshape(data[:, :, 13:23], [
                data.shape[0], data.shape[1], 5, 2])

            ball2defs = defense - ball
            ball2basket = self.RIGHT_BASKET - ball
            dotvalue = ball2defs[:, :, :, 0] * ball2basket[:, :, :,
                                                           0] + ball2defs[:, :, :, 1] * ball2basket[:, :, :, 1]
            ball2defs_len = self.__get_length(ball, defense)
            # find best defense by defense_theta
            defense_theta = np.arccos(dotvalue/(self.__get_length(
                defense, ball) * self.__get_length(self.RIGHT_BASKET, ball)+1e-3))
            defense_theta[ball2defs_len >= RADIUS] = np.inf
            best_defense_theta = np.amin(defense_theta, axis=-1)
            # clean up unused length
            valid_theta = []
            for i in range(data.shape[0]):
                valid_theta.append(
                    best_defense_theta[i, :self._length[i]].reshape([-1]))
            valid_theta = np.concatenate(valid_theta, axis=0)
            counter = np.count_nonzero(valid_theta <= THETA*np.pi/180.0)
            # show
            total_frames = data.shape[0] * data.shape[1]
            show_msg += '\'{0}\' dataset\n'.format(
                key) + '-- frequency={0:4.2f}\n'.format(counter/total_frames)
        print(show_msg)
        self._report_file.write(show_msg)

    def show_freq_heatmap(self):
        """ Vis Heat map (frequency) of positions
        """
        # mkdir
        save_path = os.path.join(self._file_name, 'heat_map')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        import matplotlib.pyplot as plt
        print('\n### show_freq_heatmap ###\n')
        for key, data in self._all_data_dict.items():
            data_len = self._length
            off_x_idx = [3, 5, 7, 9, 11]
            off_y_idx = [4, 6, 8, 10, 12]
            def_x_idx = [13, 15, 17, 19, 21]
            def_y_idx = [14, 16, 18, 20, 22]

            off_x = []
            off_y = []
            def_x = []
            def_y = []

            for epi_idx in range(data.shape[0]):
                off_x = np.append(
                    off_x, data[epi_idx, :data_len[epi_idx], off_x_idx].flatten())
                off_y = np.append(
                    off_y, data[epi_idx, :data_len[epi_idx], off_y_idx].flatten())
                def_x = np.append(
                    def_x, data[epi_idx, :data_len[epi_idx], def_x_idx].flatten())
                def_y = np.append(
                    def_y, data[epi_idx, :data_len[epi_idx], def_y_idx].flatten())

            plt.gca().set_aspect('equal', adjustable='box')
            (h, xedges, yedges, image) = plt.hist2d(def_x, def_y,
                                                    bins=[np.arange(46, 95, 1), np.arange(0, 51, 1)])
            plt.title(key+'_defense')
            plt.colorbar()
            plt.savefig(os.path.join(save_path, key+'_defense'))
            plt.close()
            if key == 'real_data':
                plt.gca().set_aspect('equal', adjustable='box')
                (h, xedges, yedges, image) = plt.hist2d(off_x, off_y,
                                                        bins=[np.arange(46, 95, 1), np.arange(0, 51, 1)])
                plt.title(key+'_offense')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, key+'_offense'))
                plt.close()

    def plot_histogram_distance_by_frames(self, mode='THETA'):
        """ plot histogram of distance frame by frame, withball and without ball
        """
        msg = '\n### plot_histogram_distance_by_frames ###\n'
        # caculate the matrix
        all_dist_dict = {}
        for key, data in self._all_data_dict.items():
            all_dist_dict[key] = self.__evalute_distance(data, mode=mode)
        # mkdir
        save_path = os.path.join(self._file_name, 'histogram_'+mode)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # vis
        all_trace = []
        for key in self._all_data_dict:
            data = self._all_data_dict[key]
            dist = all_dist_dict[key]
            # categorize into two set, withball and without ball
            ball = np.reshape(data[:, :, 0:2], [
                data.shape[0], data.shape[1], 1, 2])
            offense = np.reshape(data[:, :, 3:13], [
                data.shape[0], data.shape[1], 5, 2])
            pad_next = np.pad(offense[:, 1:], [(0, 0), (0, 1),
                                               (0, 0), (0, 0)], mode='constant', constant_values=1)
            offense_speed = self.__get_length(offense, pad_next) * self.FPS
            offense_speed[:, -1] = None
            # tally the table : if offense handle the ball
            if_handle_ball = np.empty(
                shape=[data.shape[0], data.shape[1], 5], dtype=bool)
            if_handle_ball[:] = False
            for off_idx in range(5):
                offender = offense[:, :, off_idx:off_idx+1, :]
                indices = np.where(self.__get_length(
                    offender[:, :, 0], ball[:, :, 0]) < self.WINGSPAN_RADIUS)
                if_handle_ball[indices[0], indices[1], off_idx] = True
            # clean up unused length
            for i in range(data.shape[0]):
                if_handle_ball[i, self._length[i]:] = False
            wiball_dist = dist[if_handle_ball]
            if_handle_ball = np.logical_not(if_handle_ball)
            # clean up unused length
            for i in range(data.shape[0]):
                if_handle_ball[i, self._length[i]:] = False
            woball_dist = dist[if_handle_ball]
            # # trace_wiball_dist
            bin_size = 0.5
            max_v = np.amax(wiball_dist)
            min_v = np.amin(wiball_dist)
            num_bins = int((max_v-min_v)//bin_size)+1
            counter = np.zeros(shape=[num_bins, ])
            for v in wiball_dist:
                counter[int((v-min_v)//bin_size)] += 1

            trace_wiball_dist = go.Scatter(
                name=key+'_wi_ball',
                x=[i*bin_size for i in range(num_bins)],
                y=counter
            )

            # trace_wiball_dist = go.Histogram(
            #     name=key+'_wi_ball',
            #     x=wiball_dist,
            #     autobinx=False,
            #     xbins=dict(
            #         start=0.0,
            #         end=1000.0,
            #         size=0.5
            #     ),
            #     opacity=0.5
            # )
            all_trace.append(trace_wiball_dist)
            # trace_woball_dist

            bin_size = 0.5
            max_v = np.amax(woball_dist)
            min_v = np.amin(woball_dist)
            num_bins = int((max_v-min_v)//bin_size)+1
            counter = np.zeros(shape=[num_bins, ])
            for v in woball_dist:
                counter[int((v-min_v)//bin_size)] += 1

            trace_woball_dist = go.Scatter(
                name=key+'_wo_ball',
                x=[i*bin_size for i in range(num_bins)],
                y=counter
            )
            # trace_woball_dist = go.Histogram(
            #     name=key+'_wo_ball',
            #     x=woball_dist,
            #     autobinx=False,
            #     xbins=dict(
            #         start=0.0,
            #         end=1000.0,
            #         size=0.5
            #     ),
            #     opacity=0.5
            # )
            all_trace.append(trace_woball_dist)
            msg += '{0} dataset:\n\twith ball:\tmean={1:4.2f},\tstddev={2:4.2f}\n\twithout ball:\tmean={3:4.2f},\tstddev={4:4.2f}\n'.format(
                key, np.mean(wiball_dist), np.std(wiball_dist), np.mean(woball_dist), np.std(woball_dist))
        layout = go.Layout(
            title='Distance',
            barmode='overlay',
            xaxis=dict(title='feet'),
            yaxis=dict(title='counts')
        )
        fig = go.Figure(data=all_trace, layout=layout)
        py.plot(fig, filename=os.path.join(
            save_path, 'distance_histogram.html'), auto_open=False)
        print(msg)
        self._report_file.write(msg)

    def vis_and_analysis_by_episode(self, episode_idx, mode='THETA'):
        """ given episode index, draw out the play and show all matrix result in same folder
        """
        # mkdir
        save_path = os.path.join(self._file_name, 'vis_analysis')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        length = self._length[episode_idx:episode_idx+1]
        one_episode_data = {}
        for key, data in self._all_data_dict.items():
            one_episode_data[key] = data[episode_idx:episode_idx+1]
        # vis
        video_save_path = os.path.join(
            save_path, 'videos_'+str(episode_idx))
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        # for key, data in one_episode_data.items():
        #     vis_game.plot_data(data[0], length=length[0], file_path=os.path.join(
        #         video_save_path, key+'.mp4'), if_save=True, fps=self.FPS, vis_ball_height=False, vis_annotation=False)
        # analysis
        evaluator = EvaluationMatrix(
            file_name=save_path, length=length, **one_episode_data, FPS=5, FORMULA_RADIUS=5.0)
        evaluator.plot_show_best_match(save_path=video_save_path)
        # evaluator.show_freq_of_valid_defense(RADIUS=10.0, THETA=10.0)
        # evaluator.plot_linechart_distance_by_frames(
        #     mode='THETA')
        # evaluator.show_mean_distance(mode='THETA')
        # evaluator.show_overlap_freq(OVERLAP_RADIUS=1.0, interp_flag=True)
        # evaluator.plot_histogram_vel_acc()
        # evaluator.show_freq_heatmap()
        # evaluator.plot_histogram_distance_by_frames(
        #     mode='DISTANCE')
        # evaluator.plot_histogram_distance_by_frames(
        #     mode='THETA')
        # evaluator.plot_histogram_distance_by_frames(
        #     mode='THETA_MUL_SCORE')
        # evaluator.plot_histogram_distance_by_frames(
        #     mode='THETA_ADD_SCORE')


def evaluate_new_data():
    # real_data = np.load('../data/WGAN/cnn_wi/A_real_B.npy')#TODO wrong data
    real_data = np.load('../data/WGAN/FixedFPS5.npy')[:10000:100]
    real_data = np.concatenate([real_data[:, :, 0, :3], real_data[:, :, 1:6, :2].reshape(
        [real_data.shape[0], real_data.shape[1], 10]), real_data[:, :, 6:11, :2].reshape([real_data.shape[0], real_data.shape[1], 10])], axis=-1)
    cnn_wi_2000k = np.load('../data/WGAN/cnn_wi_2000k/A_fake_B_N100.npy')[0]
    cnn_wo_368k = np.load('../data/WGAN/cnn_wo_368k/A_fake_B_N100.npy')[0]
    cnn_verify_921k = np.load(
        '../data/WGAN/cnn_verify_921k/A_fake_B_N100.npy')[0]
    # cnn_wi_2000k_not_denorm = np.load('../data/WGAN/cnn_wi_2000k_not_denorm/A_fake_B_N100.npy')[0]
    cnn_wi_mul_2000k = np.load(
        '../data/WGAN/cnn_wi_mul_2000k/A_fake_B_N100.npy')[0]
    # rnn_1000k = np.load('../data/WGAN/rnn_1000k/A_fake_B_N100.npy')[0]
    length = np.load('../data/WGAN/FixedFPS5Length.npy')[:10000:100]

    file_name = 'default'
    if os.path.exists(file_name):
        ans = input('"%s" will be removed!! are you sure (y/N)? ' % file_name)
        if ans == 'Y' or ans == 'y':
            # when not restore, remove follows (old) for new training
            shutil.rmtree(file_name)
            print('rm -rf "%s" complete!' % file_name)
    evaluator = EvaluationMatrix(
        file_name=file_name, length=length, FPS=5, FORMULA_RADIUS=5.0,
        real_data=real_data, cnn_wi_2000k=cnn_wi_2000k, cnn_wo_368k=cnn_wo_368k,
        cnn_verify_921k=cnn_verify_921k, cnn_wi_mul_2000k=cnn_wi_mul_2000k)
    evaluator.show_freq_of_valid_defense(RADIUS=10.0, THETA=10.0)
    evaluator.plot_linechart_distance_by_frames(
       mode='THETA')
    evaluator.show_mean_distance(mode='THETA')
    evaluator.show_overlap_freq(OVERLAP_RADIUS=1.0, interp_flag=False)
    evaluator.plot_histogram_vel_acc()
    evaluator.show_best_match()
    evaluator.show_freq_heatmap()
    evaluator.plot_histogram_distance_by_frames(
       mode='DISTANCE')
    evaluator.plot_histogram_distance_by_frames(
       mode='THETA')
    evaluator.plot_histogram_distance_by_frames(
       mode='THETA_MUL_SCORE')
    evaluator.plot_histogram_distance_by_frames(
       mode='THETA_ADD_SCORE')
    evaluator.vis_and_analysis_by_episode(
        episode_idx=10, mode='THETA')


if __name__ == '__main__':
    evaluate_new_data()
