# Heejin Chloe Jeong

import numpy as np

def get_action_egreedy(values, epsilon):
	if(np.random.rand() > epsilon):
		action = np.argmax(values)   					# return policy action with prob 1-epsilon
	else:
		# print("random")
		action = np.random.choice(range(len(values)))	# else return random action
	return action
	NotImplementedError

def evaluation(env, Q_table, num_itr = 10):
  """
  Semi-greedy evaluation for discrete state and discrete action spaces and an episodic environment.

  Input:
    env : an environment object. 
    Q : A numpy array. Q values for all state and action pairs. 
      Q.shape = (the number of states, the number of actions)
    step_bound : the maximum number of steps for each iteration
    num_itr : the number of iterations

  Output:
    Total number of steps taken to finish an episode (averaged over num_itr trials)
    Cumulative reward in an episode (averaged over num_itr trials)

  """
  total_step = 0 
  total_reward = 0 
  itr = 0 
  while(itr<num_itr):
    step = 0
    np.random.seed()
    state = env.reset()
    reward = 0.0
    done = False
    while(not done):
      action = get_action_egreedy(Q_table[state], 0.05)
      state_n, r, done, _ = env.step(action)
      state = state_n
      reward += r
      step +=1
    total_reward += reward
    total_step += step
    itr += 1
  env.reset()
  return total_step/float(num_itr), total_reward/float(num_itr)
