
# coding: utf-8

# In[2]:


import gym
import numpy as np
import sys
import time


# In[8]:


env = gym.make("CartPole-v0")
# env = gym.wrappers.Monitor(env, 'cartpole-experiment-1', force=True)

sumlist = []
for t in range(200):
    state = env.reset()
    i = 0
    while(True):
        i += 1
        env.render()
        action = env.action_space.sample()
        nA = env.action_space.n
        state, reward, done, _ = env.step(action)
        # print(state, action, reward)

        if done:
            print("Episode finished after {} timesteps".format(i+1))
            break
    
    sumlist.append(i)
    print("Game over...")
    
# env.monitor.close()


# In[9]:


env.close()


# In[10]:


iter_time = sum(sumlist)/len(sumlist)
print("CartPole game iter average time is: {}".format(iter_time))

