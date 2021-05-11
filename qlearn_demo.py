import matplotlib.pyplot as plt
import numpy as np
from maze_rl.maze import Maze
from maze_rl.qlearn import qlearn

def main():
    NUM_RUNS = 500
    env = Maze()
    eval_steps, eval_reward = qlearn(env=env, learning_rate=0.1, eps_n=1000, min_eps=0.01, 
                                     num_runs=NUM_RUNS, max_iter=500, pause_interval=50, discount_fact=0.98)
    # # Plot example # 
    f1, ax1 = plt.subplots() 
    ax1.plot(np.arange(0,NUM_RUNS,50),eval_steps) #repeat for different algs. 
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('steps')
    ax1.set_title('Steps to Completion')
    f2, ax2 = plt.subplots() 
    ax2.plot(np.arange(0,NUM_RUNS,50),eval_reward) #repeat for different algs.
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('reward')
    ax2.set_title('Reward Evolution')
    plt.show()

if __name__ == '__main__':
    main()