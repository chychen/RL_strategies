import gym
from bball_strategies import gym_bball
env = gym.make('bball-v0')
env.reset()
env.render()
input()

##############################################################
# from gym.envs.classic_control import rendering
# import numpy as np
# from os import path
# viewer = rendering.Viewer(940,500)
# viewer.set_bounds(0,94,0,50)
# # background img
# fname = path.join(path.dirname(__file__), "fullcourt.png")
# img = rendering.Image(fname, 94, 50)
# imgtrans = rendering.Transform(translation=(47.0, 25.0))
# img.add_attr(imgtrans)
# viewer.add_geom(img)
# # defensive players
# def_player = rendering.make_circle(radius=2.)
# def_player.set_color(0,0,1)
# def_trans = rendering.Transform(translation=(61.0, 39.0))
# def_player.add_attr(def_trans)
# viewer.add_geom(def_player)
# # offensive players
# off_player = rendering.make_circle(radius=2.)
# off_player.set_color(1,0,0)
# off_trans = rendering.Transform(translation=(60.0, 40.0))
# off_player.add_attr(off_trans)
# viewer.add_geom(off_player)
# # ball
# ball = rendering.make_circle(radius=1.)
# ball.set_color(0,1,0)
# ball_trans = rendering.Transform(translation=(60.0, 40.0))
# ball.add_attr(ball_trans)
# viewer.add_geom(ball)

# # if last_u:
# #     imgtrans.scale = (-last_u/2, np.abs(last_u)/2)
# mode='human'
# viewer.render(return_rgb_array = mode=='rgb_array')
# input()

# rod = rendering.make_capsule(1, .2)
# rod.set_color(.8, .3, .3)
# pole_transform = rendering.Transform()
# rod.add_attr(pole_transform)
# viewer.add_geom(rod)
# axle = rendering.make_circle(.05)
# axle.set_color(0,0,0)
# viewer.add_geom(axle)
# pole_transform.set_rotation(0 + np.pi/2)

#################################################

# from multiprocessing import Process

# def f(x):
#     k = 0
#     for i in range(100000):
#         k += x * x
#     print('hello world', x)
#     return x * x
    

# if __name__ == '__main__':
#     for num in range(10):
#         Process(target=f, args=[num]).start()

##################################################
# from multiprocessing import Pool
# import time

# def f(x):
#     k = 0
#     for i in range(100000):
#         k += x * x
#     return x * x


# if __name__ == '__main__':
#     p = Pool(4)
#     starttime = time.time()
#     print(p.map(f, range(1000)))
#     endtime = time.time()
#     print(endtime-starttime)
###################################################
# import tensorflow as tf

# sess = tf.InteractiveSession()

# a = tf.constant([[1,2],[3,4],[5,6]])
# indices_1 = tf.constant([0,2,1])
# indices_2 = tf.constant([0,2])

# print(a.eval())
# permuted = tf.gather(a, indices_1)
# print(permuted.eval())
# gathered = tf.gather(a, indices_2)
# print(gathered.eval())

# rows = None
# rows = tf.range(10) if rows is None else rows
# print(rows.eval())

# gather：根據一個list來取用目標
