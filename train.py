import random
from typing import TypedDict

import numpy as np
import gymnasium as gym

Q = {}
actions = (0, 1, 2, 3)


def qvalues(state):
    return [Q.get((state, a), 0) for a in actions]


def discretize(state):
    return tuple(state['agent'].tolist()), tuple(state['target'].tolist())


def probs(v, eps=1e-4):
    v = v - v.min() + eps  # add eps to avoid dividing by zero
    v /= v.sum()
    return v


Params = TypedDict('Params', {'alpha': float, 'gamma': float, 'epsilon': float})

def train(params: Params, env: gym.Env):
    Qmax = 0
    cum_rewards = []
    rewards = []
    moves = 0

    for epoch in range(100000):
        obs, _ = env.reset()
        terminated = False
        cum_reward = 0
        while not terminated:
            s = discretize(obs)
            if random.random() < params['epsilon']:
                # exploitation
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions, weights=v)[0]
            else:
                # exploration
                a = np.random.randint(env.action_space.n)  # type: ignore

            obs, rew, terminated, _, _ = env.step(a)
            cum_reward += rew  # type: ignore
            moves += 1
            if moves > 20:
                terminated = True
                moves = 0
            ns = discretize(obs)
            Q[(s, a)] = (1 - params['alpha']) * Q.get((s, a), 0) + \
                params['alpha'] * (rew + params['gamma'] * max(qvalues(ns)))

        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        if epoch % 1000 == 0:
            print(
                f"Epoch: {epoch}, Average Reward: {np.average(cum_rewards)}, alpha={params['alpha']}, epsilon={params['epsilon']}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
            cum_rewards = []
