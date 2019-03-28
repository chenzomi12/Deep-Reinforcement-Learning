
# coding: utf-8

# In[17]:


import gym
import numpy as np
import sys
import time
import pandas as pd
import matplotlib
from collections import defaultdict, namedtuple

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')


# In[4]:


env = gym.make("CartPole-v0")


# In[20]:


class QLearning():
    def __init__(self, env, num_episodes, discount=1.0, alpha=0.5, epsilon=0.1, n_bins=10):
        self.nA = env.action_space.n
        self.nS = env.observation_space.shape[0]
        self.env = env
        self.num_episodes = num_episodes
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        # Initialize Q(s; a)
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        # Keeps track of useful statistics
        record = namedtuple("Record", ["episode_lengths","episode_rewards"])
        self.rec = record(episode_lengths=np.zeros(num_episodes),
                          episode_rewards=np.zeros(num_episodes))
    
        self.cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1]
        self.pole_angle_bins = pd.cut([-2, 2], bins=n_bins, retbins=True)[1]
        self.cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1]
        self.angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins, retbins=True)[1]
        
    def __get_bins_states(self, state):
        """
        Case number of the sate is huge so in order to simplify the situation 
        cut the state sapece in to bins.
        
        if the state_idx is [1,3,6,4] than the return will be 1364
        """
        s1_, s2_, s3_, s4_ = state
        cart_position_idx = np.digitize(s1_, self.cart_position_bins)
        pole_angle_idx = np.digitize(s2_, self.pole_angle_bins)
        cart_velocity_idx = np.digitize(s3_, self.cart_velocity_bins)
        angle_rate_idx = np.digitize(s4_, self.angle_rate_bins)
        
        state_ = [cart_position_idx, pole_angle_idx, 
                  cart_velocity_idx, angle_rate_idx]
        
        state = map(lambda s: int(s), state_)
        return tuple(state)
        
    def __epislon_greedy_policy(self, epsilon, nA):

        def policy(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(self.Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy

    def __next_action(self, prob):
        return np.random.choice(np.arange(len(prob)), p=prob)

    def qlearning(self):
        """
        q-learning algo
        """
        policy = self.__epislon_greedy_policy(self.epsilon, self.nA)
        sumlist = []

        for i_episode in range(self.num_episodes):
            # Print out which episode we are on
            if 0 == (i_episode+1) % 10:
                print("\r Episode {} in {}".format(i_episode+1, self.num_episodes))
                # sys.stdout.flush()

            step = 0
            # Initialize S
            state__ = self.env.reset()
            state = self.__get_bins_states(state__)
            
            # Repeat (for each step of episode)
            while(True):
                # Choose A from S using policy derived from Q
                prob_actions = policy(state)
                action = self.__next_action(prob_actions)
                
                # Take action A, observe R, S'
                next_state__, reward, done, info = env.step(action)
                next_state = self.__get_bins_states(next_state__)
                
                # update history record
                self.rec.episode_lengths[i_episode] += reward
                self.rec.episode_rewards[i_episode] = step
                
                # TD update: Q(S; A)<-Q(S; A) + aplha*[R + discount * max Q(S'; a) − Q(S; A)]
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.discount * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

                if done:
                    # until S is terminal
                    print("Episode finished after {} timesteps".format(step))
                    sumlist.append(step)
                    break
                else:
                    step += 1
                    # S<-S'
                    state = next_state
                    
        iter_time = sum(sumlist)/len(sumlist)
        print("CartPole game iter average time is: {}".format(iter_time))
        return self.Q

cls_qlearning = QLearning(env, num_episodes=200)
Q = cls_qlearning.qlearning()


# In[27]:


from matplotlib import pyplot as plt

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths[:200])
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards[:200]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    return fig1, fig2

plot_episode_stats(cls_qlearning.rec)

