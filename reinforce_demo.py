import sys
import getopt
import matplotlib.pyplot as plt
from gym_rl.reinforce import reinforce, show_sim
import gym

def parse_cmdline():
  """ parse commandline options to configure
  """
  try:
    short_opts = 'amc:'
    long_opts = ['acrobot', 'mountaincar', 'n-eps=', 'max-steps=', 'cluster=', 'dr=', 'lr=']
    optlist, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
  except getopt.GetoptError as err:
    # print help information and exit:
    print(err)  # will print something like "option -a not recognized"
    usage()
    sys.exit(2)

  # default opts
  opts = {
    "env"           : None,
    "n_episodes"    : None,
    "max_steps"     : None,
    "discount_rate" : None,
    "learning_rate" : 0.001,
    "cluster_size"  : None,
  }

  # parse args
  for opt,arg in optlist:
    if opt in ("-a", "--acrobot"):          # default options for acrobot
      opts['env']           = 'Acrobot-v1'
      opts['n_episodes']    = 1000
      opts['max_steps']     = 500
      opts['discount_rate'] = 0.9
      opts['cluster_size']  = 1

    elif opt in ("-m", "--mountaincar"):    # default options for mountaincar
      opts['env']           = 'MountainCar-v0'
      opts['n_episodes']    = 3000
      opts['max_steps']     = 1500
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

  if (opts['env'] is None):
    print ("Environment must be specified (-a for acrobot, -m for mountaincar)")
    sys.exit()
  
  return opts


def main():
  opts = parse_cmdline()
  env = gym.make(opts.pop('env'))
  env._max_episode_steps = opts.pop('max_steps') 

  print(opts)
  scores, policy_estimator = reinforce(env, opts)

  plt.plot(scores)
  plt.xlabel('episode')
  plt.ylabel('score')
  plt.show()

  show_sim(env, policy_estimator)


if __name__ == '__main__':
  main()