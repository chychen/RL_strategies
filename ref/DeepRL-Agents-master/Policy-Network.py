
# coding: utf-8

# # Simple Reinforcement Learning in Tensorflow Part 2: Policy Gradient Method
# This tutorial contains a simple example of how to build a policy-gradient based agent that can solve the CartPole problem. For more information, see this [Medium post](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.mtwpvfi8b).
#
# For more Reinforcement Learning algorithms, including DQN and Model-based learning in Tensorflow, see my Github repo, [DeepRL-Agents](https://github.com/awjuliani/DeepRL-Agents).
#
# Parts of this tutorial are based on code by [Andrej Karpathy](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) and [korymath](https://gym.openai.com/evaluations/eval_a0aVJrGSyW892vBM04HQA).

# In[1]:

from __future__ import division

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import tensorflow as tf
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import math

try:
    xrange = xrange
except:
    xrange = range


# ### Loading the CartPole Environment
# If you don't already have the OpenAI gym installed, use  `pip install gym` to grab it.

# In[2]:

import gym
env = gym.make('CartPole-v0')


# What happens if we try running the environment with random actions? How well do we do? (Hint: not so well.)

# In[3]:

env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()


# The goal of the task is to achieve a reward of 200 per episode. For every step the agent keeps the pole in the air, the agent recieves a +1 reward. By randomly choosing actions, our reward for each episode is only a couple dozen. Let's make that better with RL!

# ### Setting up our Neural Network agent
# This time we will be using a Policy neural network that takes observations, passes them through a single hidden layer, and then produces a probability of choosing a left/right movement. To learn more about this network, see [Andrej Karpathy's blog on Policy Gradient networks](http://karpathy.github.io/2016/05/31/rl/).

# In[4]:

# hyperparameters
H = 10  # number of hidden layer neurons
batch_size = 5  # every how many episodes to do a param update?
# feel free to play with this to train faster or more stably.
learning_rate = 1e-2
gamma = 0.99  # discount factor for reward

D = 4  # input dimensionality


# In[5]:

tf.reset_default_graph()

# This defines the network as it goes from taking an observation of the environment to
# giving a probability of chosing to the action of moving left or right.
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# The loss function. This sends the weights in the direction of making actions
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loglik = tf.log(input_y * (input_y - probability) +
                (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Our optimizer
# Placeholders to send the final gradients through when we update.
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


# ### Advantage function
# This function allows us to weigh the rewards our agent recieves. In the context of the Cart-Pole task, we want actions that kept the pole in the air a long time to have a large reward, and actions that contributed to the pole falling to have a decreased or negative reward. We do this by weighing the rewards from the end of the episode, with actions at the end being seen as negative, since they likely contributed to the pole falling, and the episode ending. Likewise, early actions are seen as more positive, since they weren't responsible for the pole falling.

# In[ ]:

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# ### Running the Agent and Environment

# Here we run the neural network agent, and have it act in the CartPole environment.

# In[ ]:

xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()  # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        # Rendering the environment slows things down,
        # so let's only look at it once our agent is doing a good job.
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation, [1, D])

        # Run the policy network and get an action to take.
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)  # observation
        y = 1 if action == 0 else 0  # a "fake label"
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # record reward (has to be done after we call step() to get reward for previous action)
        drs.append(reward)

        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr //= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads, feed_dict={
                             observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={
                         W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * \
                    0.99 + reward_sum * 0.01
                print('Average reward for episode %f.  Total average reward %f.' % (
                    reward_sum // batch_size, running_reward // batch_size))

                if reward_sum // batch_size > 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(episode_number, 'Episodes completed.')


# As you can see, the network not only does much better than random actions, but achieves the goal of 200 points per episode, thus solving the task!

# In[ ]:


# In[ ]:
