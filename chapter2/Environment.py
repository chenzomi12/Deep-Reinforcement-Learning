
# coding: utf-8

# In[49]:


import numpy as np
import sys
from six import StringIO, b
from pprint import PrettyPrinter
get_ipython().run_line_magic('pprint', '')

from gym import utils
from gym.envs.toy_text import discrete


# In[60]:


pp = PrettyPrinter(indent=2)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MAPS = {'4x4':["SOOO","OXOX","OOOX","XOOG"]}


# In[61]:


class GridworldEnv(discrete.DiscreteEnv):
    """
    FrozenLakeEnv1 is a copy environment from GYM toy_text FrozenLake-01

    You are an agent on an 4x4 grid and your goal is to reach the terminal
    state at the bottom right corner.
    
    For example, a 4x4 grid looks as follows:
    
    S  O  O  O
    O  X  O  X
    O  O  O  X
    X  O  O  G
    
    S : starting point, safe
    O : frozen surface, safe
    X : hole, fall to your doom
    G : goal, where the frisbee is located
    
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    """
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, desc=None, map_name='4x4'):
        self.desc = desc = np.asarray(MAPS[map_name], dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.shape = desc.shape
        
        nA = 4                    # 动作集个数
        nS = np.prod(desc.shape)  # 状态集个数

        MAX_Y = desc.shape[0]
        MAX_X = desc.shape[1]

        # initial state distribution [ 1.  0.  0.  ...] 
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()
        
        P = {}          
        state_grid = np.arange(nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] == [(probability, nextstate, reward, done), ...]
            P[s] = {a : [] for a in range(nA)}

            s_letter = desc[y][x]
            is_done = lambda letter: letter in b'GX'
            reward = 0.0 if s_letter in b'G' else -1.0
            
            if is_done(s_letter):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                sl_up = desc[ns_up//MAX_Y][ns_up%MAX_X]
                sl_right = desc[ns_right//MAX_Y][ns_right%MAX_X]
                sl_down = desc[ns_down//MAX_Y][ns_down%MAX_X]
                sl_left = desc[ns_left//MAX_Y][ns_left%MAX_X]
                
                P[s][UP] = [(1.0, ns_up, reward, is_done(sl_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(sl_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(sl_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(sl_left))]
                
            it.iternext()
                
        self.P = P
        
        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close: # 初始化环境Environment的时候不显示
            return
        
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        
        state_grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(state_grid, flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
        
            # 对于当前状态用红色标注
            if self.s == s:
                desc[y][x] = utils.colorize(desc[y][x], "red", highlight=True)
            
            it.iternext()
       
        outfile.write("\n".join(' '.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
        
env = GridworldEnv()


# In[62]:


observation = env.reset()
for _ in range(5):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("action:{}({})".format(action, ["Up","Right","Down","Left"][action]))
    print("done:{}, observation:{}, reward:{}".format(done, observation, reward))
    if done:
        pp.pprint(env.P)
        print("Episode finished after {} timesteps".format(_+1))
        break

