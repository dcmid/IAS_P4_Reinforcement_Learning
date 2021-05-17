import numpy as np
import torch
import pickle as pkl
from torch import nn
from torch import optim
from time import sleep


# Neural Net using pytorch -------------------------------------------------------------------------------------

class MyNN():
  """ Linear Function Approximator (neural net) used to estimate policy
  """
  def __init__(self, env, activation_func = nn.GELU()):
    self.n_inputs = env.observation_space.shape[0]  # number of inputs equal to length of observation vector
    self.n_outputs = env.action_space.n             # number of outputs equal to length of action vector

    self.layers = nn.Sequential(
      nn.Linear(self.n_inputs,64),
      activation_func,
      nn.Linear(64,self.n_outputs),
      nn.Softmax(dim=-1)
    )

  def predict(self, state, return_tensor=True):
    """ takes np state space, runs forward pass, and returns result as np """
    action_probs = self.layers(torch.FloatTensor(state))
    if return_tensor:
      return action_probs
    else:
      return action_probs.detach().numpy()

# Helper Functions ------------------------------------------------------------------------------------

def apply_discount(rewards, discount_rate):
  """ calculates expected rewards for each state, accounting for future rewards and discount rate
  """
  discount = [discount_rate**t for t in range(len(rewards))]
  discounted_rewards = np.multiply(rewards, discount)
  expected_rewards = np.cumsum(discounted_rewards[::-1])[::-1]  # reversed cumsum to add all future discounted rewards
  return expected_rewards

def get_loss(log_probs, rewards):
  """ computes loss from one episode
  """
  losses = -torch.stack(log_probs)*torch.FloatTensor(rewards)
  total_loss = losses.sum()
  return total_loss


# REIFORCE ----------------------------------------------------------------------------------------------------

def reinforce(env, opts):
  """ executes REINFORCE on given environment with specified options
  """
  n_episodes = opts['n_episodes']
  discount_rate = opts['discount_rate']
  learning_rate = opts['learning_rate']
  cluster_size = opts['cluster_size']

  policy_estimator = MyNN(env)
  optimizer = optim.Adam(policy_estimator.layers.parameters(),lr=learning_rate)
  actions = np.arange(env.action_space.n) # all possible actions
  
  total_rewards = []

  for n in range(n_episodes):
    state = env.reset()
    states = []
    rewards = []
    log_probs = []
    done = False
    step = 0
    while not done:                                                       # loop until episode is done
      if step % cluster_size == 0:
        p_actions = policy_estimator.predict(state)                       # get probability vector for actions
        action = np.random.choice(actions, p=p_actions.detach().numpy())  # sample from actions w/ probability vector
        log_prob = torch.log(p_actions[action])                           # calculate log prob of action that was taken
      n_state, reward, done, _ = env.step(action)                         # execute action on environment
      
      states.append(state)
      rewards.append(reward)
      log_probs.append(log_prob)

      state = n_state
      step += 1

    total_rewards.append(sum(rewards))        

    disc_rewards = apply_discount(rewards, discount_rate)           # calculate discounted rewards
    disc_rewards_norm = (disc_rewards - np.mean(disc_rewards))\
                                /np.std(disc_rewards)               # normalize to std & zero mean

    optimizer.zero_grad()
    loss = get_loss(log_probs,disc_rewards_norm)
    loss.backward()
    optimizer.step()

    if n % 100 == 0:
      print(f"Episode:    {n}")
      print(f"Mean Score: {np.mean(total_rewards[-100:])}")

  return [total_rewards, policy_estimator]

# Display model in environment

def show_sim(env, policy_estimator):
  state = env.reset()
  actions = np.arange(env.action_space.n) # all possible actions
  done = False
  while not done:
    p_actions = policy_estimator.predict(state)                       # get probability vector for actions
    action = np.random.choice(actions, p=p_actions.detach().numpy())  # sample from actions w/ probability vector
    state, _, done, _ = env.step(action)                              # execute action on environment
    env.render()
    sleep(0.1)
