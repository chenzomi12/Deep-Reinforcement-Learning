
# coding: utf-8

# In[1]:

import numpy as np
import pprint
from Environment import GridworldEnv
from pprint import PrettyPrinter

get_ipython().magic('pprint')
pp = PrettyPrinter(indent=4)


# In[43]:

def calc_action_value(state, V, discount_factor=1.0):
        """
        Calculate the expected value of each action in a given state.
        对于给定的状态 s 计算其动作 a 的期望值
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    
def value_iteration(env, theta=0.1, discount_factor=1.0):
    """
    Value Iteration Algorithm. 值迭代算法
    """
    # 初始化状态值
    V = np.zeros(env.nS)

    # 迭代计算找到最优的状态值函数 optimal value function
    for _ in range(50):
        delta = 0 # 停止标志位
        
        # 计算每个状态的状态值
        for s in range(env.nS):
            A = calc_action_value(s, V) # 执行一次找到当前状态的动作期望
            best_action_value = np.max(A) # 选择最好的动作期望作为新的状态值
            
            # 计算停止标志位
            delta = max(delta, np.abs(best_action_value - V[s])) 
            
            # 更新状态值函数
            V[s] = best_action_value  
            
        if delta < theta:
            break
    
    
    # 输出最优策略：通过最优状态值函数找到决定性策略
    policy = np.zeros([env.nS, env.nA]) # 初始化策略
    
    for s in range(env.nS):
        # 执行一次找到当前状态的最优状态值的动作期望 A
        A = calc_action_value(s, V)
        
        # 选出状态值最大的作为最优动作
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    
    return policy, V


# In[44]:

env = GridworldEnv()
policy, v = value_iteration(env)

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# The real policy Ieration function following
# =============

# In[ ]:



