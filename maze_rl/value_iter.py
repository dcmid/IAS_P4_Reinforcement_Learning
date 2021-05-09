import numpy as np
from .maze import *

def update_value(env, value, state, discount_fact=0.9):
    """ gives value of each possible action from current state """
    actions = [0,1,2,3]   # [u d l r]
    slip_prob = env.slip  # probability of slipping

    max_val = 0
    for a in actions:
        move_reward, move_state, _ = env.step(state,a,slip_ena=False)           # action successfully taken
        slip_reward, slip_state, _ = env.step(state,ACTMAP[a],slip_ena=False)   # action slipped
        move_val = value[move_state]                                            # value if action taken
        slip_val = value[slip_state]                                            # value if slipped

        action_reward = slip_prob * slip_reward + (1-slip_prob) * move_reward   # expected reward from action
        action_val    = slip_prob * slip_val + (1-slip_prob) * move_val         # expected value from action

        new_val = action_reward + discount_fact * action_val                    # sum of reward and value scaled by discount
        if (new_val > max_val):
            max_val = new_val

    return max_val

def value_iteration(env=Maze(), cutoff=0.1, discount_fact=0.9):
    #env = Maze()
    value = np.zeros(env.snum)

    max_change = cutoff + 1  # this is just temporary so that the loop starts
    while(max_change > cutoff):
        max_change = 0
        for s in range(env.snum): # for all possible states
            v = value[s]
            value[s] = update_value(env, value, s, discount_fact)
            change = np.abs(v - value[s])
            if change > max_change:
                max_change = change

    # generate policy
    actions = [0,1,2,3]   # [u d l r]
    policy = np.zeros((env.snum,env.anum))
    for s in range(len(policy)):
        n_states = [env.step(s,a,slip_ena=False)[1] for a in actions]   # possible next states
        n_vals   = [value[n_s] for n_s in n_states]                     # values of next states
        best_action = np.argmax(n_vals)                                 # action to most valuable state
        policy[s,best_action] = 1
    return policy