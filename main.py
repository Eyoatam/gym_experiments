import argparse

import random
import numpy as np

from world import GridWorldEnv
from train import train, extract_obs, qvalues, probs


parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
parser.add_argument(
    "--alpha",
    type=float,
    default=0.30,
    metavar="A",
    help="learning rate (default: 0.30)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor (default: 0.99)",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.90,
    metavar="E",
    help="exploration factor (default: 0.90)",
)
parser.add_argument("--render", type=bool, default=False, help="render the environment")

args = parser.parse_args()


def main():
    train_env = GridWorldEnv()
    train(
        params={"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon},
        env=train_env,
    )

    actions = (0, 1, 2, 3)
    # run simulation
    if args.render:
        test_env = GridWorldEnv(render_mode="human")
    else:
        test_env = train_env
    print("-" * 60)
    for _ in range(10):
        obs, _ = test_env.reset()
        terminated = False
        while not terminated:
            s = extract_obs(obs)
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions, weights=v)[0]
            obs, _, terminated, _, _ = test_env.step(a)
    test_env.close()


if __name__ == "__main__":
    main()
