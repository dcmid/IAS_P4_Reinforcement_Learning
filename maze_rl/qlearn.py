import numpy as np
import matplotlib.pyplot as plt 
from .maze import *
from evaluation import * 


def update_q(q_table, state, action, n_state, reward, learning_rate=0.5, discount_fact=0.9):
    curr_q = q_table[state, action]
    next_q = (reward + discount_fact * np.max(q_table[n_state]))
    new_q = (1- learning_rate) * curr_q + learning_rate * scaled_next_q
    return new_q


def qlearn(learning_rate=0.5, eps_decay=0.001, min_eps=0.01, max_epochs=100, max_iter=1000):
    eval_steps , eval_reward = [], []
    env = Maze() 
    state = env.reset()

    q_table = np.zeros((env.snum,env.anum))
    done = False
    while not done:
        action = get_action_egreedy(q_table[state], epsilon)



        avg_step, avg_reward = evaluation(env, q_table) 
        eval_steps.append(avg_step) 
        eval_reward.append(avg_reward)


# # Plot example # 
# f1, ax1 = plt.subplots() 
# ax1.plot(np.arange(0,5000,50),eval_steps) #repeat for different algs. 
# f2, ax2 = plt.subplots() 
# ax2.plot(np.arange(0,5000,50),eval_reward) #repeat for different algs.