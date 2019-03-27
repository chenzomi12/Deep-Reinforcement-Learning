
# coding: utf-8

# In[24]:


import numpy as np
import gym
from termcolor import colored


# In[31]:


env = gym.make("Blackjack-v0")

def show_state(state):
    player, dealer, ace = state
    dealer = sum(env.dealer)
    print("Player:{}, ace:{}, Dealer:{}".format(player, ace, dealer))

def simple_strategy(state):
    player, dealer, ace = state
    return 0 if player >= 18 else 1

def episode(num_episodes):
    episode = []
    for i_episode in range(5):
        print("\n" + "="* 30)
        state = env.reset()
        for t in range(10):
            show_state(state)
            action = simple_strategy(state)
            action_ = ["STAND", "HIT"][action]
            print("Player Simple strategy take action:{}".format(action_))
            
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                show_state(state)
                # [-1(loss), -(push), 1(win)]
                reward_ = ["loss", "push", "win"][int(reward+1)]
                print("Game {}.(Reward {})".format(reward_, int(reward)))
                print("PLAYER:{}\t DEALER:{}".format(colored(env.player, 'red'), 
                                                     colored(env.dealer, 'green')))
                break
                
            state = next_state

episode(1000)

