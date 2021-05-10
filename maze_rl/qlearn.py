import numpy as np
import matplotlib.pyplot as plt 
from .maze import *
from evaluation import * 


def update_q(q_table, state, action, n_state, reward, learning_rate=0.5, discount_fact=0.9):
    curr_q = q_table[state, action]
    next_q = (reward + discount_fact * np.max(q_table[n_state]))
    new_q = (1- learning_rate) * curr_q + learning_rate * scaled_next_q
    return new_q

def update_epsilon(epsilon, state_count, eps_n, min_eps):
    n_epsilon = eps_n / (eps_n + state_count)
    return max(n_epsilon, min_eps)


def qlearn(env=Maze(), learning_rate=0.5, eps_n=100, min_eps=0.01, max_runs=100, max_iter=1000):
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
    epoch = 0
    training_done = False
    while not training_done:
        i = 0
        run_done = False
        while (not run_done) and (i < max_iter):
            state_counts[state] += 1                                                        # update state count of current state
            action = get_action_egreedy(q_table[state], epsilon)                            # get action
            n_state, reward, run_done = env.step(action)                                    # execute action and get reward/next state
            q_table = update_q(q_table, state, action, n_state, reward, learning_rate=0.5)  # update q-table
            i += 1



        avg_step, avg_reward = evaluation(env, q_table) 
        eval_steps.append(avg_step) 
        eval_reward.append(avg_reward)


# # Plot example # 
# f1, ax1 = plt.subplots() 
# ax1.plot(np.arange(0,5000,50),eval_steps) #repeat for different algs. 
# f2, ax2 = plt.subplots() 
# ax2.plot(np.arange(0,5000,50),eval_reward) #repeat for different algs.