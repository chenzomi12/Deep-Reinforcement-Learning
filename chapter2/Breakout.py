
# coding: utf-8

# In[1]:


import gym
import time
from gym import wrappers


# In[2]:


env = gym.make('Breakout-v0')
env = wrappers.Monitor(env, '/Users/chenzomi')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        
        time.sleep(2)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

