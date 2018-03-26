"""
EvaluationMatrix:
- show_mean_distance()
    Mean/Stddev of distance between offense (wi/wo ball) and defense
- show_overlap_freq()
    Overlap frequency (judged by threshold = radius 1 ft)
- plot_histogram_vel_acc()
    Histogram of DEFENSE's velocity and acceleration. (mean,stddev)
- Vis Heat map (frequency) of positions
- Best match between real and defense’s position difference.(mean,stddev)
- Compare to formula (defense sync with offense movement)
- plot_linechart_distance_by_frames():
    Vis dot distance of each offense to closest defense frame by frame 
    (with indicators: inside 3pt line, ball handler, paint area)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import colorsys


class EvaluationMatrix(object):
    """ EvaluationMatrix
    """

    def __init__(self, length=None, **kwargs):
        """ all data in keargs must have the shape=(num_episode, length, 23)

        Args
        ----
        kwargs : all items' values should have the same shape as real_data.shape
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
        self.WEIGHT = 10

        self._all_data_dict = kwargs
        self._length = length
        self._num_episodes = self._length.shape[0]

    def show_overlap_freq(self, OVERLAP_RADIUS=1.0):
        """ Overlap frequency (judged by threshold = OVERLAP_RADIUS)
        """
        print('Overlap Frequency')
        for key, data in self._all_data_dict.items():
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
                    temp_len[i, self._length[i]:] = np.inf
                counter += np.count_nonzero(temp_len <= OVERLAP_RADIUS)
            # show
            show_msg = '\'{}\' dataset\n'.format(
                key) + '-- frequency={}\n'.format(counter/total_frames)
            print(show_msg)

    def show_mean_distance(self, mode='THETA'):
        """ Mean/Stddev of distance between offense (wi/wo ball) and defense
        """
        print('Compare Distance with/without ball')
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
            show_msg = '\'{}\' dataset\n'.format(key) + '-- wi ball: mean={}, stddev={}\n'.format(
                wi_mean, wi_std) + '-- wo ball: mean={}, stddev={}\n'.format(wo_mean, wo_std)
            print(show_msg)

    def __get_length(self, a, b, axis=-1):
        """ get distance between a and b by axis
        """
        vec = a-b
        return np.sqrt(np.sum(vec*vec, axis=axis))

    def __get_visual_aid(self, data):
        """ return markers
        """
        ball = np.reshape(data[:, :, 0:2], [
            data.shape[0], data.shape[1], 1, 2])
        offense = np.reshape(data[:, :, 3:13], [
            data.shape[0], data.shape[1], 5, 2])
        pad_next = np.pad(offense[:, 1:], [(0, 0), (0, 1),
                                           (0, 0), (0, 0)], mode='constant', constant_values=1)
        offense_speed = self.__get_length(offense, pad_next)
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
        offense_speed = self.__get_length(offense, pad_next)
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

        # clean up unused length
        for i in range(data.shape[0]):
            dist[i, self._length[i]:] = 0.0

        return dist

    def plot_linechart_distance_by_frames(self, file_name='default', mode='THETA'):
        """ Vis dot distance of each offense to closest defense frame by frame 
        (with indicators: inside 3pt line, ball handler, paint area)

        Args
        ----
        handler_idx : int, shape=[num_episodes, padded_length, 5]
            one hot vector represent the ball handler idx for each frame
        if_inside_3pt : 
        if_inside_paint : 
        """
        # caculate the matrix
        all_dist_dict = {}
        for key, data in self._all_data_dict.items():
            all_dist_dict[key] = self.__evalute_distance(data, mode=mode)
        all_marker_dict = self.__get_visual_aid(
            self._all_data_dict['real_data'])
        # mkdir
        save_path = os.path.join(file_name, 'linechart_'+mode)
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

    def plot_histogram_vel_acc(self, file_name='default'):
        """ Histogram of DEFENSE's speed and acceleration. (mean,stddev)
        """
        # mkdir
        save_path = os.path.join(file_name, 'histogram')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        all_trace_speed = []
        all_trace_acc = []
        for key, data in self._all_data_dict.items():
            defense = np.reshape(data[:, :, 13:23], [
                data.shape[0], data.shape[1], 5, 2])
            speed = self.__get_length(defense[:, 1:], defense[:, :-1])
            acc = self.__get_length(speed[:, 1:], speed[:, :-1])
            # clean up unused length
            valid_speed = []
            valid_acc = []
            for i in range(data.shape[0]):
                valid_speed.append(
                    speed[i, :self._length[i]-1].reshape([-1, ]))
                valid_acc.append(acc[i, :self._length[i]-2].reshape([-1, ]))
            trace_speed = go.Histogram(
                name=key,
                x=np.concatenate(valid_speed, axis=0),
                opacity=0.5
            )
            all_trace_speed.append(trace_speed)
            trace_acc = go.Histogram(
                name=key,
                x=np.concatenate(valid_acc, axis=0),
                opacity=0.5
            )
            all_trace_acc.append(trace_acc)

        layout_speed = go.Layout(
            title='Speed',
            barmode='overlay',
            xaxis=dict(title='speed (feet/sec)'),
            yaxis=dict(title='counts')
        )
        layout_acc = go.Layout(
            title='Acceleration',
            barmode='overlay',
            xaxis=dict(title='acceleration (feet/sec^2)'),
            yaxis=dict(title='counts')
        )
        fig_speed = go.Figure(data=all_trace_speed, layout=layout_speed)
        fig_acc = go.Figure(data=all_trace_acc, layout=layout_acc)
        py.plot(fig_speed, filename=os.path.join(save_path,'speed_histogram.html'), auto_open=False)
        py.plot(fig_acc, filename=os.path.join(save_path,'acc_histogram.html'), auto_open=False)


def main():
    fake_data = np.load('../data/WGAN/A_fake_B.npy')
    real_data = np.load('../data/WGAN/A_real_B.npy')
    length = np.load('../data/WGAN/len.npy')
    evaluator = EvaluationMatrix(
        length=length, real_data=real_data, fake_data=fake_data)
    # evaluator.plot_linechart_distance_by_frames(file_name='default', mode='THETA')
    # evaluator.show_mean_distance(mode='THETA')
    # evaluator.show_overlap_freq(OVERLAP_RADIUS=1.0)
    evaluator.plot_histogram_vel_acc()


if __name__ == '__main__':
    main()
