# IAS Project 4 - Reinforcement Learning
Link to GitHub repo: https://github.com/dcmid/IAS_P4_Reinforcement_Learning

This repository includes package support and demo file 
for each of the 3 challenges in this project. Challenges 
1 and 2 use the *maze_rl* package. Challenge 3 uses the *gym_rl* package.

The demos can be executed directly from the root directory.

## 1. Value Iteration
To execute demo:
```
python value_iter_demo.py <policy output file>
```
If no file is specified, the policy is not saved.

One example output policy from running value iteration 
can be found in the file *policy.npy* in the root directory.

## 2. Q-Learning
To execute demo:
```
python qlearn_demo.py
```

## 3. Continuous State Space

### REINFORCE
To execute demo:
```
python reinforce_demo.py -m (for mountaincar)
python reinforce_demo.py -a (for acrobot)
```

### Discretized Q-Learning
To execute demo:
```
python gym_qlearn_demo.py -m (for mountaincar)
python qym_qlearn_demo.py -a (for acrobot)
```

## Packages
The code for this project is organized into two packages, *maze_rl* and *gym_rl*, corresponding 
to the Maze and OpenAI Gym problems, respectively.
### maze_rl
Code for value iteration is located in the *value_iter.py* module.

Code for q-learning is located in the *qlearn.py* module.

### gym_rl
Code for REINFORCE is located in the *reinforce.py* module.

Code  for q-learning is located in the *qlearn.py* module.