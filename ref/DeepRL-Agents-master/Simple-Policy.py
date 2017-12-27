
# coding: utf-8

# # Simple Reinforcement Learning in Tensorflow Part 1: 
# ## The Multi-armed bandit
# This tutorial contains a simple example of how to build a policy-gradient based agent that can solve the multi-armed bandit problem. For more information, see this [Medium post](https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149).
# 
# For more Reinforcement Learning algorithms, including DQN and Model-based learning in Tensorflow, see my Github repo, [DeepRL-Agents](https://github.com/awjuliani/DeepRL-Agents). 

# In[1]:

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# ### The Bandit
# Here we define our bandit. For this example we are using a four-armed bandit. The pullBandit function generates a random number from a normal distribution with a mean of 0. The lower the bandit number, the more likely a positive reward will be returned. We want our agent to learn to always choose the arm that will give that positive reward.

# In[2]:

#List out our bandit arms. 
#Currently arm 4 (index #3) is set to most often provide a positive reward.
bandit_arms = [0.2,0,-0.2,-2]
num_arms = len(bandit_arms)
def pullBandit(bandit):
    #Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return 1
    else:
        #return a negative reward.
        return -1


# ### The Agent
# The code below established our simple neural agent. It consists of a set of values for each of the bandit arms. Each value is an estimate of the value of the return from choosing the bandit. We use a policy gradient method to update the agent by moving the value for the selected action toward the recieved reward.

# In[3]:

tf.reset_default_graph()

#These two lines established the feed-forward part of the network. 
weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network.
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_output = tf.slice(output,action_holder,[1])
loss = -(tf.log(responsible_output)*reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)


# ### Training the Agent

# We will train our agent by taking actions in our environment, and recieving rewards. Using the rewards and actions, we can know how to properly update our network in order to more often choose actions that will yield the highest rewards over time.

# In[4]:

total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_arms) #Set scoreboard for bandit arms to 0.

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        
        #Choose action according to Boltzmann distribution.
        actions = sess.run(output)
        a = np.random.choice(actions,p=actions)
        action = np.argmax(actions == a)

        reward = pullBandit(bandit_arms[action]) #Get our reward from picking one of the bandit arms.
        
        #Update the network.
        _,resp,ww = sess.run([update,responsible_output,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        
        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward))
        i+=1
print("\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
    print("...and it was right!")
else:
    print("...and it was wrong!")


# In[ ]:



