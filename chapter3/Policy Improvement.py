
# coding: utf-8

# In[3]:


import numpy as np
import pprint
from Environment import GridworldEnv
from pprint import PrettyPrinter

get_ipython().run_line_magic('pprint', '')
pp = PrettyPrinter(indent=4)


# In[15]:


env = GridworldEnv()


# In[16]:


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
        
        # 停止标志位
        if delta <= theta:
            break
    
    return np.array(V)


# In[21]:


def policy_improvement(env, policy, discount_factor=1.0):
    """
    Policy Imrpovement.
    Iterativedly evaluates and improves a policy until an 
    optimal policy is found or to the limited iter threshold.
    
    Args:
        env: the environment.
        policy_eval_fun: Policy Evaluation function with 3 
        argements: policy, env, discount_factor.
        
    Returns:
        tuple(policy, V).
    """
    k = 0
    while True:
        print(k)
        V = policy_eval(policy, env, discount_factor)
        print("random policy:\n", policy)
        print("policy eval:\n",V.reshape(env.shape))
        policy_stable = True
        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
                    if done and next_state != 15:
                        action_values[a] = float('-inf')

            print("action_values:\n",s, action_values)
            
            best_a = np.argmax(action_values)
            
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        print("policy\n", np.reshape(np.argmax(policy, axis=1), env.shape))
            
        if policy_stable:
            return policy, V
        k+=1


# In[22]:


random_policy = np.ones([env.nS, env.nA])/env.nA
policy, v = policy_improvement(env, random_policy)

print("\nReshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# The real policy Ieration function following
# =============

# In[ ]:


def policy_iteration(env, policy, discount_factor=1.0):
    while True:
        # 评估当前策略 policy
        V = policy_eval(policy, env, discount_factor)

        # policy 标志位，当某状态的策略更改后该标志位为 False
        policy_stable = True
        
        # 策略改进
        for s in range(env.nS):
            # 在当前状态和策略下选择概率最高的动作
            old_action = np.argmax(policy[s])
            
            # 在当前状态和策略下找到最优动作
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
                    if done and next_state != 15:
                        action_values[a] = float('-inf')

            print("action_values:\n",s, action_values)
            
            # 采用贪婪算法更新当前策略
            best_action = np.argmax(action_values)
            
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        
        # 选择的动作不再变化，则代表策略已经稳定下来
        if policy_stable:
            # 返回最优策略和对应状态值
            return policy, V

