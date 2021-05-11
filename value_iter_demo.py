#!/usr/bin/python

import sys
from maze_rl.maze import *
from maze_rl.value_iter import value_iteration

def main(save_path=None):
    env = Maze()
    policy = value_iteration(env=env, cutoff=0.1, discount_fact=0.9)    

    if save_path is not None:
        np.save(save_path,policy)
        
    run_maze(policy=policy, env=env)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()