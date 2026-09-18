"""
train.py
--------
Trains an ADPRL agent (DDPG core) on the CEC environment, following
Algorithm 1 of the paper.

Two target policies are trained, exactly as described in Sec. 6.1:
    - LO (Latency-Optimized):  lambda_t = 1.0, lambda_e = 0.0
    - EE (Energy-Efficient):   lambda_t = 0.5, lambda_e = 0.5

"Asynchronous" data collection (server=1, workers=4) is approximated here
by round-robin stepping through `--workers` independent CECEnv instances
(each with its own freshly generated workload) every training iteration,
all pushing transitions into one shared replay buffer, and one shared
actor/critic pair being updated after every worker step -- this reproduces
the *qualitative* behaviour of Algo. 1 (asynchronous rollouts + shared
learner) without requiring real multiprocessing / distributed workers,
which keeps the script simple to run on a single GPU.

Usage
-----
    python train.py --objective LO --episodes 2000 --tasks-per-episode 20 \
        --edge-nodes 25 --out runs/lo_model.pt

    python train.py --objective EE --episodes 2000 --tasks-per-episode 20 \
        --edge-nodes 25 --out runs/ee_model.pt

See README.md for the full set of flags and recommended settings to
reproduce the paper's experiments (Sec. 6).
"""

from __future__ import annotations
import argparse
import os
import time
import numpy as np
import torch

from dag_generator import DAGTaskGenerator
from edge_env import NetworkModel, CECEnv
from ddpg_agent import ADPRLAgent


def make_env(num_edge_nodes, lambda_t, lambda_e, seed=None):
    net = NetworkModel(num_edge_nodes, seed=seed)
    env = CECEnv(net, lambda_t=lambda_t, lambda_e=lambda_e)
    return env


def run_episode(env, gen, agent, num_tasks, explore=True, train=True):
    tasks = gen.generate_batch(num_tasks)
    state = env.reset(tasks)
    done = False
    ep_reward = 0.0
    steps = 0
    while not done:
        action_tensor = agent.act(state, explore=explore)
        action = agent.decode_action(action_tensor)
        next_state, reward, done, info = env.step(action)
        if train:
            agent.store(state, action_tensor, reward, next_state, done)
            agent.update()
        state = next_state
        ep_reward += reward
        steps += 1
    act, ec, orr = env.episode_metrics()
    return ep_reward, steps, act, ec, orr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", choices=["LO", "EE"], default="LO")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--tasks-per-episode", type=int, default=20)
    p.add_argument("--edge-nodes", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=2e-3)
    p.add_argument("--gamma", type=float, default=0.001,
                    help="reward decay; paper reports 0.001 (see ddpg_agent.py docstring)")
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=str, default="runs/model.pt")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    lambda_t, lambda_e = (1.0, 0.0) if args.objective == "LO" else (0.5, 0.5)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # one env per async worker, each with an independent random topology seed
    envs = [
        make_env(args.edge_nodes, lambda_t, lambda_e, seed=args.seed + w)
        for w in range(args.workers)
    ]
    gens = [
        DAGTaskGenerator(args.edge_nodes, seed=args.seed + 100 + w)
        for w in range(args.workers)
    ]

    agent = ADPRLAgent(
        state_dim=envs[0].state_dim,
        num_nodes=args.edge_nodes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"[ADPRL] device={agent.device}, objective={args.objective} "
          f"(lambda_t={lambda_t}, lambda_e={lambda_e})")

    history = {"episode": [], "reward": [], "ACT": [], "EC": [], "OR": []}
    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        w = ep % args.workers
        env, gen = envs[w], gens[w]
        ep_reward, steps, act, ec, orr = run_episode(
            env, gen, agent, args.tasks_per_episode, explore=True, train=True
        )
        history["episode"].append(ep)
        history["reward"].append(ep_reward)
        history["ACT"].append(act)
        history["EC"].append(ec)
        history["OR"].append(orr)

        if ep % args.log_every == 0:
            recent = slice(max(0, ep - args.log_every), ep)
            print(
                f"ep {ep:5d}/{args.episodes}  "
                f"reward={np.mean(history['reward'][recent]):8.3f}  "
                f"ACT={np.mean(history['ACT'][recent]):7.2f}  "
                f"EC={np.mean(history['EC'][recent]):7.2f}  "
                f"OR={np.mean(history['OR'][recent]):5.2f}  "
                f"elapsed={time.time() - t0:6.1f}s"
            )

    agent.save(args.out)
    print(f"[ADPRL] saved trained model to {args.out}")

    # also dump training curve for plotting
    import json
    hist_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"[ADPRL] saved training history to {hist_path}")


if __name__ == "__main__":
    main()
