
# coding: utf-8

# In[2]:


import numpy as np
import pprint
from Environment import GridworldEnv
from pprint import PrettyPrinter

get_ipython().run_line_magic('pprint', '')
pp = PrettyPrinter(indent=4)


# In[32]:


env = GridworldEnv()
random_policy = np.ones([env.nS, env.nA])/env.nA
print(random_policy)


# In[41]:


def policy_eval(policy, environment, discount_factor=1.0, theta=1.0):
    env = environment # 环境变量
    
    # 初始化一个全0的价值函数
    V = np.zeros(env.nS)
    
    # 迭代开始
    for _ in range(10000):
        delta = 0
        
        # 对于GridWorld中的每一个状态都进行全备份
        for s in range(env.nS):
            v = 0
            # 检查下一个有可能执行的动作
            for a, action_prob in enumerate(policy[s]):
                
                # 对于每一个动作检查下一个状态
                for  prob, next_state, reward, done in env.P[s][a]:
                    # 累积计算下一个动作的期望价值
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
                    
            # 选出最大的变化量
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        
        print("="*60, _)
        print(V.reshape(env.shape))
        
        # 停止标志位
        if delta <= theta:
            break
    
    return np.array(V)


# In[6]:


v = policy_eval(random_policy, env)
print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# In[21]:


def gen_random_policy():
    return np.random.choice(4, size=((16)))

n_policy = 100
policy_pop = [gen_random_policy() for _ in range(n_policy)]
pprint.pprint(policy_pop[:10])


# # Test the Spesfiy Input Policy
# 
# if I input a speciay policy what the value funciton calcuate by the policy evalutaiton.

# In[9]:


input_policy = [2,1,2,3,2,0,2,0,1,2,2,0,0,1,1,0]

env = GridworldEnv()
policy = np.zeros([env.nS, env.nA])

for _, x in enumerate(input_policy):
    policy[_][x] = 1
    
print(policy)


# In[7]:


def policy_eval(policy, environment, discount_factor=1.0, theta=0.1):
    env = environment # 环境变量
    
    # 初始化一个全0的价值函数
    V = np.zeros(env.nS)
    
    # 迭代开始
    for _ in range(50):
        delta = 0
        
        # 对于GridWorld中的每一个状态都进行全备份
        for s in range(env.nS):
            v = 0
            # 检查下一个有可能执行的动作
            for a, action_prob in enumerate(policy[s]):
                
                # 对于每一个动作检查下一个状态
                for  prob, next_state, reward, done in env.P[s][a]:
                    # 累积计算下一个动作的期望价值
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
                    
            # 选出最大的变化量
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        
        print("="*60, _)
        print(V.reshape(env.shape))
        
        # 停止标志位
        if delta <= theta:
            break
    
    return np.array(V)


# In[8]:


v = policy_eval(policy, env)
print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

