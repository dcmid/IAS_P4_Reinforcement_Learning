import numpy as np
import matplotlib.pyplot as plt 
from .maze import *
from .evaluation import * 


def update_q(q_table, state, action, n_state, reward, learning_rate=0.5, discount_fact=0.9):
    curr_q = q_table[state, action]
    next_q = (reward + discount_fact * np.max(q_table[n_state]))
    new_q = (1- learning_rate) * curr_q + learning_rate * next_q
    return new_q

def get_epsilon(state_count, eps_n, min_eps):
    n_epsilon = eps_n / (eps_n + state_count)
    return max(n_epsilon, min_eps)


def qlearn(env=Maze(), learning_rate=0.5, eps_n=1000, min_eps=0.01, num_runs=100, max_iter=5000, pause_interval=50, discount_fact=0.9):
    """ executes q learning on environment
    
        Args:
            env:            environment in which actions are simulated and q-learning implemented
            learning rate:  rate at which q table changes due to new information
            eps_n:          constant that determines rate of epsilon decay
                            epsilon(s) = eps_n/(eps_n + visits(s))
            min_eps:        minimum epsilon value. epsilon decay ignored below this
            max_runs:       maximum number of runs to be executed
            max_iter:       max iterations without finishing before run exits


     """
    eval_steps , eval_reward = [], []
    state = env.reset()

    state_counts = np.zeros(env.snum)       # number of times each state has been visited
    q_table = np.zeros((env.snum,env.anum))
    i = 0
    for run in range(num_runs):
        run_done = False
        while (not run_done):
            epsilon = get_epsilon(state_counts[state], eps_n, min_eps)                          # get epsilon for current state
            print(epsilon)
            action = get_action_egreedy(q_table[state], epsilon)                                # get action
            reward, n_state, run_done = env.step(state, action)                                 # execute action and get reward/next state
            q_table[state,action] = update_q(q_table, state, action, n_state, reward, 
                                             learning_rate=0.5, discount_fact=discount_fact)   # update q-table
            
            state_counts[state] += 1                                                            # update state count
            state = n_state                                                                     # update state
            i += 1
            if (i >= max_iter):                                                                 # stop run if max_iter reached
                run_done = True

        if (run_done == True):  # if run done, reset iter and state
            i = 0
            state = env.reset()

        if (run % pause_interval == 0):
            avg_step, avg_reward = evaluation(env, q_table) 
            eval_steps.append(avg_step) 
            eval_reward.append(avg_reward)
    return [eval_steps, eval_reward]