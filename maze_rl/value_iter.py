import numpy as np
from maze import *

def update_value(env, state, discount_fact=0.9):
    actions = [0,1,2,3] # [u d l r]
    action_vals = [env.step(s,a)[0] for a in actions]
    return np.argmax(action_vals)

def value_iteration():
    env = Maze()
    value = np.zeros(env.snum)

    change = 0
    for s in range(env.snum): # for all possible states
        v = value[s]
        value[s] = update_value(env, s)
