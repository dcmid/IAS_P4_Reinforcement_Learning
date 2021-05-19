import gym
from gym.spaces import Discrete
import numpy as np

from .evaluation import evaluation, get_action_egreedy
from maze_rl.qlearn import get_epsilon, update_q

class DiscreteObsWrapper(gym.ObservationWrapper):
  """ Converts Box observation to single integer

      Ref: https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html
  """
  def __init__(self, env, nbins=10):
    super().__init__(env)
    
    low_bound = self.observation_space.low
    high_bound = self.observation_space.high

    self.nbins = nbins
    self.bin_bounds = [np.linspace(l, h, nbins + 1) for l, h in           # create list w/ ceiling for obs that fall
                       zip(low_bound.flatten(), high_bound.flatten())]    # into each bin for every obs
    self.observation_space = Discrete(nbins ** self.observation_space.shape[0])
    
  def _to_one_number(self, obs):
    return sum([(o - 1) * (self.nbins ** i) for i, o in enumerate(obs)])

  def observation(self, obs):
    digits = [np.digitize([x], bins)[0]
              for x, bins in zip(obs.flatten(), self.bin_bounds)]
    return self._to_one_number(digits)


def qlearn(env, opts, eps_n=1000, min_eps=0.01):
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
  n_episodes = opts['n_episodes']
  discount_rate = opts['discount_rate']
  learning_rate = opts['learning_rate']
  cluster_size = opts['cluster_size']
  pause_interval = opts['pause_interval']

  eval_steps , eval_reward = [], []
  state = env.reset()

  n_states = env.observation_space.n
  n_actions = env.action_space.n

  state_counts = np.zeros(n_states)       # number of times each state has been visited
  q_table = np.zeros((n_states,n_actions))
  i = 0
  for run in range(n_episodes):
    run_done = False
    while (not run_done):
      epsilon = get_epsilon(state_counts[state], eps_n, min_eps)                          # get epsilon for current state
      # print(epsilon)
      if i % cluster_size == 0:
        action = get_action_egreedy(q_table[state], epsilon)                              # get action
      n_state, reward, run_done, _ = env.step(action)                                   # execute action and get reward/next state
      q_table[state,action] = update_q(q_table, state, action, n_state, reward, 
                                        learning_rate=learning_rate, discount_fact=discount_rate)   # update q-table
      
      state_counts[state] += 1                                                            # update state count
      state = n_state                                                                     # update state
      i += 1

    if (run_done == True):  # if run done, reset iter and state
      i = 0
      state = env.reset()

    if (run % pause_interval == 0):
      # print(q_table)
      avg_step, avg_reward = evaluation(env, q_table)
      eval_steps.append(avg_step)
      eval_reward.append(avg_reward)
      print(f"Episode: {run}")
      # print(avg_step)
      print(avg_reward)
  return [eval_steps, eval_reward]