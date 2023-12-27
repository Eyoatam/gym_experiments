from world import GridWorldEnv  
import random
import numpy as np

def create_world(size=5, render_mode=None):
    env = GridWorldEnv(size=size, render_mode=render_mode)
    return env

env = create_world(size=5, render_mode=None)

# Q-Learning

Q = {}
actions = (0, 1, 2, 3)

def qvalues(state):
    return [Q.get((state,  a), 0) for a in actions]

def discretize(state):
    return tuple(state["agent"].tolist()), tuple(state["target"].tolist())
# hyperparameters
alpha = 0.3
gamma = 0.9
epsilon = 0.90

def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v

def train():
    Qmax = 0
    cum_rewards = []
    rewards = []
    moves = 0

    for epoch in range(1000):
        obs, _ = env.reset()
        terminated = False
        cum_reward = 0
        while not terminated:
            s = discretize(obs)
            if random.random() < epsilon:
                # exploitation
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions, weights=v)[0]
            else:
                # exploration
                a = np.random.randint(env.action_space.n)

            obs, rew, terminated, _, _  = env.step(a)
            cum_reward += rew
            moves += 1
            if moves > 20:
                terminated = True
                moves = 0
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))

        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        if epoch%10==0:
            print(f"Epoch: {epoch}, Average Reward: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]

train()

# run simulation
env = create_world(size=5, render_mode="human")
print("-"*60)
# print(obs)
for i in range(10):
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        s = discretize(obs)
        v = probs(np.array(qvalues(s)))
        a = random.choices(actions, weights=v)[0]
        obs, _, terminated, _, _ = env.step(a)
env.close()

