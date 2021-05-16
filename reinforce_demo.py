import matplotlib.pyplot as plt
from gym_rl.reinforce import reinforce, show_sim
import gym

env = gym.make('Acrobot-v1')

scores, policy_estimator = reinforce(env)

plt.plot(scores)
plt.show()

show_sim(env, policy_estimator)