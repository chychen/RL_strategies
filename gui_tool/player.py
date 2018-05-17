
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import codecs
import random
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Ellipse, Color
from kivy.properties import NumericProperty, ObjectProperty, StringProperty
from kivy.config import Config
from kivy.uix.popup import Popup
import kivy
kivy.require('1.9.0')
Config.set('graphics', 'width', '1800')
Config.set('graphics', 'height', '800')


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Cursor(Widget):
    """
    The customed widget to draw cursor on the SlideBar.
    """

    def __init__(self, *args, **kwargs):
        super(Cursor, self).__init__(*args, **kwargs)


class SlideBar(Widget):
    """
    The customed widget to draw SlideBar on the canvas.
    """
    # the height of the SlideBar
    y_offset = NumericProperty(30.0)

    def __init__(self, *args, **kwargs):
        super(SlideBar, self).__init__(*args, **kwargs)


class BBallCourt(Widget):
    """
    """

    def __init__(self, *args, **kwargs):
        super(BBallCourt, self).__init__(*args, **kwargs)
        self.FPS = 5.0
        self.vis_factor_x = 0.0
        self.vis_factor_y = 0.0
        self.ball_r = 0.0
        self.player_r = 0.0
        self.all_players = []
        self.last_pos = None
        self.bind(width=self.redraw, height=self.redraw)

    def redraw(self, *args):
        if self.last_pos is not None:
            self.update_players(self.last_pos)

    def update_players(self, players_pos):
        """
        players_pos : shape=(11, 2)
            ball, off*5, def*5
        """
        self.last_pos = players_pos
        self.vis_factor_x = 1.0/94.0*self.width
        self.vis_factor_y = 1.0/50.0*self.height
        self.ball_r = 0.4
        self.player_r = 1.25
        if len(self.all_players) == 0:
            with self.canvas.after:
                # offense
                Color(1.0, 0.0, 0.0, 1)
                for i in range(5):
                    self.all_players.append(
                        Ellipse(pos=[0, 0], size=[self.player_r*2*self.vis_factor_x, self.player_r*2*self.vis_factor_y]))
                # defense
                Color(0.0, 0.0, 1.0, 1)
                for i in range(5):
                    self.all_players.append(
                        Ellipse(pos=[0, 0], size=[self.player_r*2*self.vis_factor_x, self.player_r*2*self.vis_factor_y]))
                # ball, draw last, show in the front
                Color(0.0, 1.0, 0.0, 1)
                self.all_players = [
                    Ellipse(pos=[0, 0], size=[self.ball_r*2*self.vis_factor_x, self.ball_r*2*self.vis_factor_y])] + self.all_players
        
        for i, pos in enumerate(players_pos):
            if i == 0:  # ball
                self.all_players[i].pos = [
                    (pos[0]-self.ball_r)*self.vis_factor_x, (pos[1]-self.ball_r)*self.vis_factor_y]
                self.all_players[i].size = [self.ball_r*2*self.vis_factor_x, self.ball_r*2*self.vis_factor_y]
            else:  # players
                self.all_players[i].pos = [
                    (pos[0]-self.player_r)*self.vis_factor_x, (pos[1]-self.player_r)*self.vis_factor_y]
                self.all_players[i].size = [self.player_r*2*self.vis_factor_x, self.player_r*2*self.vis_factor_y]


class AppEngine(FloatLayout):
    """
    main app
    """
    court = ObjectProperty(None)
    frame_idx = ObjectProperty(None)
    # buttons
    last_button = ObjectProperty(None)
    play_pause_button = ObjectProperty(None)
    reset_button = ObjectProperty(None)
    next_button = ObjectProperty(None)
    # Label
    label = ObjectProperty(None)
    frame_idx_str = StringProperty("None")
    episode_len_str = StringProperty("None")
    # scroll bar
    slide_bar = ObjectProperty(None)
    frame_cursor = ObjectProperty(None)
    # engine status
    is_init = False

    def __init__(self, *args, **kwargs):
        super(AppEngine, self).__init__(*args, **kwargs)
        # hide
        self.label.opacity = 0.0
        self.court.opacity = 0.0
        self.slide_bar.opacity = 0.0
        self.frame_cursor.opacity = 0.0
        self.last_button.opacity = 0.0
        self.play_pause_button.opacity = 0.0
        self.reset_button.opacity = 0.0
        self.next_button.opacity = 0.0
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file",
                            content=content, size_hint=(0.5, 0.5),
                            auto_dismiss=False)
        self._popup.open()

    def load(self, path, filename):
        # show
        self.label.opacity = 1.0
        self.court.opacity = 1.0
        self.slide_bar.opacity = 1.0
        self.frame_cursor.opacity = 1.0
        self.last_button.opacity = 1.0
        self.play_pause_button.opacity = 1.0
        self.reset_button.opacity = 1.0
        self.next_button.opacity = 1.0
        self.is_init = True
        filepath = os.path.join(path, filename[0])
        # data to vis
        data = np.load(filepath)
        self.episode = data[0]
        assert self.episode.shape[1:] == tuple((11, 2)), "episode shape[1:] {} doesn't match shape [11,2]".format(self.episode.shape[1:])
        self.episode_len = self.episode.shape[0]
        self.is_playing = False
        self.frame_idx = 0
        self.court.update_players(self.episode[self.frame_idx])
        # label
        self.frame_idx_str = str(self.frame_idx)
        self.episode_len_str = str(self.episode_len-1)
        # bindings
        self.last_button.bind(on_release=self.last_button_callback)
        self.play_pause_button.bind(on_release=self.play_pause_callback)
        self.reset_button.bind(on_release=self.reset_button_callback)
        self.next_button.bind(on_release=self.next_button_callback)
        self.bind(frame_idx=self.update_court)
        # playing event
        self.playing_evnet = Clock.schedule_interval(
            self.playing_callback, 1.0/self.court.FPS)
        self.playing_evnet.cancel()
        self._popup.dismiss()

    def dismiss_popup(self):
        self._popup.dismiss()

    def update_court(self, *args):
        self.court.update_players(self.episode[self.frame_idx])
        # update label
        self.frame_idx_str = str(self.frame_idx)
        # update cursor
        unit_len = self.width/(self.episode_len-1)
        self.frame_cursor.x = self.frame_idx * unit_len - 5.0

    def next_button_callback(self, instance):
        if self.is_init:
            if self.frame_idx < self.episode_len-1:
                self.frame_idx += 1
            # pause
            if self.is_playing:
                self.playing_evnet.cancel()
                self.is_playing = not self.is_playing

    def last_button_callback(self, instance):
        if self.is_init:
            if self.frame_idx > 0:
                self.frame_idx -= 1
            # pause
            if self.is_playing:
                self.playing_evnet.cancel()
                self.is_playing = not self.is_playing

    def play_pause_callback(self, instance):
        if self.is_init:
            self.playpause_action()

    def playpause_action(self):
        if self.is_init:
            if not self.is_playing and self.frame_idx < self.episode_len-1:
                self.playing_evnet()
                self.is_playing = not self.is_playing
            else:
                self.playing_evnet.cancel()
                self.is_playing = not self.is_playing

    def reset_button_callback(self, instance):
        self.frame_idx = 0

    def playing_callback(self, dt):
        if self.frame_idx < self.episode_len-1:
            self.frame_idx += 1
            self.court.update_players(self.episode[self.frame_idx])
        else:
            self.playing_evnet.cancel()
            self.is_playing = not self.is_playing

    def on_touch_move(self, touch):
        super(AppEngine, self).on_touch_down(touch)
        if self.is_init:
            self.touch_action(touch, mode='on_touch_move')

    def on_touch_down(self, touch):
        super(AppEngine, self).on_touch_down(touch)
        if self.is_init:
            self.touch_action(touch, mode='on_touch_down')

    def touch_action(self, touch, mode):
        # touch on bball court == play pr pause
        if touch.x >= 0.0 and touch.x < self.slide_bar.width and touch.y >= 0.0 and touch.y < self.slide_bar.height:
            # scrallable bar area
            # according to episode length, which discretize the position of cursor
            unit_len = self.width/(self.episode_len-1)
            self.frame_idx = int((touch.x+unit_len/2) // unit_len)
        elif touch.x >= self.center_x - self.court.width/2 and touch.x < self.center_x + self.court.width/2 and touch.y >= self.center_y - self.court.height/2 and touch.y < self.center_y + self.court.height/2:
            # court area
            if mode == 'on_touch_down':
                self.playpause_action()
        else:
            pass


class PlayerApp(App):
    """
    main app builder
    """

    def build(self):
        app = AppEngine()
        return app


if __name__ == '__main__':
    PlayerApp().run()
