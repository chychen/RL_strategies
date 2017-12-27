
# coding: utf-8

# # Simple Reinforcement Learning with Tensorflow: Part 0 - Q-Tables
# In this iPython notebook we implement a Q-Table algorithm that solves the FrozenLake problem. To learn more, read here: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# 
# For more reinforcment learning tutorials, see:
# https://github.com/awjuliani/DeepRL-Agents

# In[1]:

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# ### Load the environment

# In[2]:

env = gym.make('FrozenLake-v0')


# ### Implement Q-Table learning algorithm

# In[3]:

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)


# In[4]:

print("Score over time: " +  str(sum(rList)/num_episodes))


# In[5]:

print("Final Q-Table Values")
print(Q)


# In[ ]:



