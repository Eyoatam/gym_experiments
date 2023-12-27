import random
import numpy as np

from world import GridWorldEnv  
from train import train, discretize, qvalues, probs

train_env = GridWorldEnv()
train(params={'alpha': 0.3, 'gamma': 0.9, 'epsilon': 0.90}, env=train_env)

actions = (0, 1, 2, 3)
# run simulation
test_env = GridWorldEnv(render_mode='human')
print("-"*60)
for i in range(10):
    obs, _ = test_env.reset()
    terminated = False
    while not terminated:
        s = discretize(obs)
        v = probs(np.array(qvalues(s)))
        a = random.choices(actions, weights=v)[0]
        obs, _, terminated, _, _ = test_env.step(a)
test_env.close()

