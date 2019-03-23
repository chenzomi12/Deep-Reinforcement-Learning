
# coding: utf-8

# In[8]:


import numpy as np
import random


# In[17]:


def epsilon_greedy(nA, R, T, epsilon=0.6):
    """
    输入：
       nA 动作数量
       R 奖励函数
       T 迭代次数
    """
    # 初始化累积奖励 r
    r = 0          
    count = [0]*nA
    
    for _ in range(T):
        if np.random.rand() < epsilon:
            # 探索：以均匀分布随机选择
            a = np.random.randint(q_value.shape[0])
        else:
            # 利用：选择价值函数最大的动作
            a = np.argmax(q_value[:])
        
        # 更新累积奖励和价值函数
        v = R(a)
        r = r + v
        q_value[a] = (q_value[a] * count[a] + v)/(count[a]+1)
        count[a] += 1
        
    return r

