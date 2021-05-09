import numpy as np
from maze import *

def update_value(env, state, discount_fact=0.9):
    """ gives value of each possible action from current state """
    actions = [0,1,2,3]   # [u d l r]s]
    slip_prob = env.slip  # probability of slipping

    move_reward = [env.step(s,a)[0] for a in actions]                       # expected reward if action successfully taken
    slip_reward = [env.step(s,ACTMAP[a])[0] for a in actions]               # expected reward if slips
    action_reward = slip_prob * slip_reward + (1-slip_prob) * act_reward    # expected reward from action attempt
    action_val = discount_fact * action_reward                              # scale by discount factor

    return np.argmax(action_vals)

def value_iteration(cutoff=0.1, discount_fact=0.9):
    env = Maze()
    value = np.zeros(env.snum)

    max_change = cutoff + 1  # this is just temporary so that the loop starts
    while(max_change > cutoff):
        max_change = 0
        for s in range(env.snum): # for all possible states
            v = value[s]
            value[s] = update_value(env, s, discount_fact)
            max_change = np.max(max_change, np.abs(v - value[s]))
