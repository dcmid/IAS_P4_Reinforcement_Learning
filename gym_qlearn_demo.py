import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym_rl.qlearn import qlearn, DiscreteObsWrapper

def parse_cmdline():
  """ parse commandline options to configure
  """
  try:
    short_opts = 'amc:'
    long_opts = ['acrobot', 'mountaincar', 'n-eps=', 'max-steps=', 'cluster=', 'dr=', 'lr=', 'pi=']
    optlist, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
  except getopt.GetoptError as err:
    # print help information and exit:
    print(err)  # will print something like "option -a not recognized"
    usage()
    sys.exit(2)

  # default opts
  opts = {
    "env"             : None,
    "n_episodes"      : None,
    "max_steps"       : None,
    "discount_rate"   : None,
    "learning_rate"   : None,
    "cluster_size"    : None,
    "pause_interval"  : 50,
  }

  # parse args
  for opt,arg in optlist:
    if opt in ("-a", "--acrobot"):          # default options for acrobot
      opts['env']           = 'Acrobot-v1'
      opts['n_episodes']    = 2000
      opts['max_steps']     = 500
      opts['learning_rate'] = 0.1
      opts['discount_rate'] = 0.9
      opts['cluster_size']  = 1

    elif opt in ("-m", "--mountaincar"):    # default options for mountaincar
      opts['env']           = 'MountainCar-v0'
      opts['n_episodes']    = 1000
      opts['max_steps']     = 1000
      opts['learning_rate'] = 0.1
      opts['discount_rate'] = 0.98
      opts['cluster_size']  = 10

    elif opt == '--n-eps':                  # custom number of episodes
      opts['n_episodes'] = int(arg)

    elif opt == '--max-steps':              # custom max episode steps
      opts['max_steps'] = int(arg)

    elif opt == '--dr':                     # custon discount rate
      opts['discount_rate'] = float(arg)

    elif opt == '--lr':                     # custon learning rate
      opts['learning_rate'] = float(arg)

    elif opt in ("-c", "--cluster"):        # custom cluster size
      opts['cluster_size'] = int(arg)
      
    elif opt == "--pi":                     # pause interval
      opts['pause_interval'] = int(arg)

  if (opts['env'] is None):
    print ("Environment must be specified (-a for acrobot, -m for mountaincar)")
    sys.exit()
  
  return opts

def main():
  opts = parse_cmdline()
  env = gym.make(opts.pop('env'))
  env._max_episode_steps = opts.pop('max_steps') 
  d_env = DiscreteObsWrapper(env)
  
  steps, reward = qlearn(d_env, opts)

  xs = opts['pause_interval'] * np.arange(len(reward))

  plt.plot(xs,reward)
  plt.title('Score Evolution')
  plt.xlabel('episode')
  plt.ylabel('average reward')
  plt.show()

if __name__ == '__main__':
  main()