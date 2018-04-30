# read tensor directly from the graph, then we could inference both actions and value
# and the returned action will interact with env in cotroller scripts to get observation -> fake samples to train with discriminator

import tensorflow as tf
from bball_strategies.networks import gail_ppo_nets


# observation space = shape=(batch_size, episode_length, 10, 14, 2)
obs_dummy = tf.zeros(shape=(1, 1, 5, 14, 2))
# action space = shape=(batch, episode_length, 23)


class PPOPolicy(object):

    def __init__(self, config, env):
        with tf.name_scope('ppo_generator'):
            self._obs = tf.placeholder(dtype=tf.float32, shape=[
                                       None, None]+list(env.observation_space.shape))
            dict_ = gail_ppo_nets.gail_def_gaussian(
                config, None, self._obs, None, reuse=True)

            self._act_sample = dict_.policy[0].sample()
            self._act_mode = dict_.policy[0].mode()

    def act(self, observation, stochastic=True):
        """
        Args
        ----
        observation : float, shape=[batch_size, episode_length]+observation_space
            observation_space = [buffer_size, 14, 2]

        Retrun
        ------
        act : float, shape=[batch_size,]+ action_space
            for defense -> action_space=[10,]
        """
        feed_dict = {
            self._obs: observation
        }
        if stochastic:
            return tf.get_default_session().run(self._act_sample[0, 0], feed_dict)
        else:
            return tf.get_default_session().run(self._act_mode[0, 0], feed_dict)


def main():
    """ demo code
    """
    import agents
    import gym
    import numpy as np
    from bball_strategies.scripts.gail import configs
    from bball_strategies.gym_bball import tools

    class MonitorWrapper(gym.wrappers.Monitor):
        # init_mode 0 : init by default
        def __init__(self, env, init_mode=None, if_vis_trajectory=False, if_vis_visual_aid=False, init_positions=None, init_ball_handler_idx=None):
            super(MonitorWrapper, self).__init__(env=env, directory='./test/',
                                                 video_callable=lambda count: count % 1 == 0, force=True)
            env.init_mode = init_mode
            env.if_vis_trajectory = if_vis_trajectory
            env.if_vis_visual_aid = if_vis_visual_aid
            env.init_positions = init_positions
            env.init_ball_handler_idx = init_ball_handler_idx

    config = agents.tools.AttrDict(configs.default())
    env = gym.make(config.env)
    env = MonitorWrapper(env,
                         init_mode=3,  # init from dataset in order
                         if_vis_trajectory=False,
                         if_vis_visual_aid=True)
    obs = env.reset()
    ppo_policy = PPOPolicy(config, env)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(230):
            act = ppo_policy.act(np.array(obs)[None, None])
            transformed_act = [
                int(0),  # Discrete(3) must be int
                [0, 0],  # Box(2,)
                np.zeros(shape=[5, 2]),  # Box(5, 2)
                np.reshape(act, [5, 2])  # Box(5, 2)
            ]
            obs, reward, done, info = env.step(transformed_act)
            env.render()
            if done:
                env.reset()
                env.render()


if __name__ == '__main__':
    main()
