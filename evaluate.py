"""
evaluate.py
-----------
Loads trained ADPRL (LO and/or EE) checkpoints and compares them against
the four baselines (Random, LE, Greedy, DQN+FCFS) while sweeping the same
parameters explored in the paper's evaluation (Sec. 6):

    (a) number of tasks            [10, 20, 30, 40]
    (b) number of subtasks/task    [25, 50, 75, 100]
    (c) average link bandwidth     [2, 4, 6, 8]  Mbps
    (d) average processing speed   [10, 20, 30, 40] Mcps
    (e) number of edge nodes       [25, 50, 75, 100]

For each configuration it reports Average Completion Time (ACT), total
Energy Consumption (EC), and Offloading Ratio (OR), matching the metrics
in Sec. 6.1 and Figures 6, 8, 9, 10.

Outputs:
    - a CSV of all raw results
    - PNG plots analogous to Fig. 9 (ACT) and Fig. 10 (EC) for each swept
      parameter

Usage
-----
    python evaluate.py --lo-model runs/lo_model.pt --ee-model runs/ee_model.pt \
        --out-dir results/

NOTE: because DQN+FCFS is itself learned online, this script trains a
fresh short-lived DQN+FCFS instance per configuration for a warm-up number
of episodes before measuring it, so that comparisons are not against an
untrained network. Increase --dqn-warmup for closer-to-paper convergence
(at the cost of runtime).
"""

from __future__ import annotations
import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dag_generator import DAGTaskGenerator
from edge_env import NetworkModel, CECEnv
from ddpg_agent import ADPRLAgent
from baselines import RandomPolicy, LocalExecutionPolicy, GreedyPolicy, DQNFCFSPolicy


def build_env(num_nodes, bw_mean, speed_mean, lambda_t, lambda_e, seed):
    net = NetworkModel(
        num_nodes, proc_speed_mean=speed_mean, bandwidth_mean=bw_mean, seed=seed
    )
    return CECEnv(net, lambda_t=lambda_t, lambda_e=lambda_e)


def run_policy_episode(env, gen, policy, num_tasks, is_adprl=False, explore=False):
    tasks = gen.generate_batch(num_tasks)
    state = env.reset(tasks)
    done = False
    while not done:
        if is_adprl:
            action_tensor = policy.act(state, explore=explore)
            action = policy.decode_action(action_tensor)
        else:
            action = policy.act(state, env)
        next_state, reward, done, info = env.step(action)
        if hasattr(policy, "observe"):
            policy.observe(state, action, reward, next_state, done)
        state = next_state
    return env.episode_metrics()  # (ACT, EC, OR)


def evaluate_config(num_tasks, num_subtasks_range, num_nodes, bw_mean, speed_mean,
                     adprl_lo, adprl_ee, seed, eval_episodes=5, dqn_warmup=50, skip_dqn=False):
    results = {}

    # --- ADPRL (LO / EE) ---
    for tag, agent, (lt, le) in [
        ("ADPRL (LO)", adprl_lo, (1.0, 0.0)),
        ("ADPRL (EE)", adprl_ee, (0.5, 0.5)),
    ]:
        if agent is None:
            continue
        if agent.num_nodes != num_nodes:
            # ADPRL's actor output size is fixed at training time (one
            # softmax head per edge node). To evaluate it on a different
            # node count you must train a dedicated model for that count
            # (see README "Varying the number of edge nodes"). We skip
            # (NaN) rather than silently truncating the action space.
            continue
        env = build_env(num_nodes, bw_mean, speed_mean, lt, le, seed)
        gen = DAGTaskGenerator(num_nodes, subtask_range=num_subtasks_range, seed=seed + 1)
        acts, ecs, ors = [], [], []
        for _ in range(eval_episodes):
            a, e, o = run_policy_episode(env, gen, agent, num_tasks, is_adprl=True, explore=False)
            acts.append(a); ecs.append(e); ors.append(o)
        results[tag] = (np.mean(acts), np.mean(ecs), np.mean(ors))

    # --- Random / LE / Greedy ---
    for tag, PolicyCls in [("Random", RandomPolicy), ("LE", LocalExecutionPolicy), ("Greedy", GreedyPolicy)]:
        env = build_env(num_nodes, bw_mean, speed_mean, 0.5, 0.5, seed)
        gen = DAGTaskGenerator(num_nodes, subtask_range=num_subtasks_range, seed=seed + 2)
        policy = PolicyCls(num_nodes) if PolicyCls is RandomPolicy else PolicyCls()
        acts, ecs, ors = [], [], []
        for _ in range(eval_episodes):
            a, e, o = run_policy_episode(env, gen, policy, num_tasks, is_adprl=False)
            acts.append(a); ecs.append(e); ors.append(o)
        results[tag] = (np.mean(acts), np.mean(ecs), np.mean(ors))

    # --- DQN+FCFS (warmed up quickly then measured) ---
    if skip_dqn:
        results["DQN+FCFS"] = (np.nan, np.nan, np.nan)
    else:
        env = build_env(num_nodes, bw_mean, speed_mean, 0.5, 0.5, seed)
        gen = DAGTaskGenerator(num_nodes, subtask_range=num_subtasks_range, seed=seed + 3)
        dqn = DQNFCFSPolicy(env.state_dim, num_nodes)
        for _ in range(dqn_warmup):
            run_policy_episode(env, gen, dqn, num_tasks, is_adprl=False)
        dqn.eps = 0.02
        acts, ecs, ors = [], [], []
        for _ in range(eval_episodes):
            a, e, o = run_policy_episode(env, gen, dqn, num_tasks, is_adprl=False)
            acts.append(a); ecs.append(e); ors.append(o)
        results["DQN+FCFS"] = (np.mean(acts), np.mean(ecs), np.mean(ors))

    return results


def sweep_and_plot(param_name, values, fixed, adprl_lo, adprl_ee, out_dir, seed=0,
                    x_values=None, cfg_key=None, eval_episodes=5, dqn_warmup=50, skip_dqn=False):
    """`values` are used as the x-axis labels; `cfg_key`/each value determine
    what actually gets overridden in the env config (useful when the x-axis
    label differs from the raw config value, e.g. subtasks-per-task)."""
    algos = ["Random", "LE", "Greedy", "DQN+FCFS", "ADPRL (LO)", "ADPRL (EE)"]
    act_curves = {a: [] for a in algos}
    ec_curves = {a: [] for a in algos}
    or_curves = {a: [] for a in algos}
    x_values = x_values if x_values is not None else values
    cfg_key = cfg_key or param_name

    for v in values:
        cfg = dict(fixed)
        cfg[cfg_key] = v
        res = evaluate_config(
            cfg["num_tasks"], cfg["num_subtasks_range"], cfg["num_nodes"],
            cfg["bw_mean"], cfg["speed_mean"], adprl_lo, adprl_ee, seed,
            eval_episodes=eval_episodes, dqn_warmup=dqn_warmup, skip_dqn=skip_dqn,
        )
        for a in algos:
            if a in res:
                act_curves[a].append(res[a][0])
                ec_curves[a].append(res[a][1])
                or_curves[a].append(res[a][2])
            else:
                act_curves[a].append(np.nan)
                ec_curves[a].append(np.nan)
                or_curves[a].append(np.nan)

    os.makedirs(out_dir, exist_ok=True)

    # CSV dump
    csv_path = os.path.join(out_dir, f"sweep_{param_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", param_name, "ACT", "EC", "OR"])
        for a in algos:
            for v, act, ec, orr in zip(x_values, act_curves[a], ec_curves[a], or_curves[a]):
                w.writerow([a, v, act, ec, orr])

    # ACT plot (Fig. 9 style)
    plt.figure(figsize=(5, 4))
    for a in algos:
        plt.plot(x_values, act_curves[a], marker="o", label=a)
    plt.xlabel(param_name)
    plt.ylabel("Average Completion Time")
    plt.title(f"ACT vs {param_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ACT_vs_{param_name}.png"), dpi=150)
    plt.close()

    # EC plot (Fig. 10 style)
    plt.figure(figsize=(5, 4))
    for a in algos:
        plt.plot(x_values, ec_curves[a], marker="o", label=a)
    plt.xlabel(param_name)
    plt.ylabel("Energy Consumption")
    plt.title(f"EC vs {param_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"EC_vs_{param_name}.png"), dpi=150)
    plt.close()

    print(f"[eval] {param_name}: wrote {csv_path} and plots to {out_dir}")
    return act_curves, ec_curves, or_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lo-model", type=str, default=None)
    p.add_argument("--ee-model", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="smaller sweeps for a fast sanity check")
    p.add_argument("--eval-episodes", type=int, default=5,
                    help="episodes averaged per algorithm per config (lower = faster, noisier)")
    p.add_argument("--dqn-warmup", type=int, default=50,
                    help="episodes used to (re)train DQN+FCFS from scratch per config; "
                         "this dominates evaluate.py's runtime -- lower it a lot for a quick look "
                         "(e.g. 5-10), keep it higher (50-200) for a fairer DQN+FCFS comparison")
    p.add_argument("--skip-dqn", action="store_true",
                    help="skip the DQN+FCFS baseline entirely (biggest time saver)")
    args = p.parse_args()

    base_nodes = 25
    fixed = dict(num_tasks=20, num_subtasks_range=(25, 25), num_nodes=base_nodes,
                 bw_mean=10.0, speed_mean=40.0)

    adprl_lo = adprl_ee = None
    if args.lo_model:
        adprl_lo = ADPRLAgent(state_dim=10, num_nodes=base_nodes)
        adprl_lo.load(args.lo_model)
    if args.ee_model:
        adprl_ee = ADPRLAgent(state_dim=10, num_nodes=base_nodes)
        adprl_ee.load(args.ee_model)

    task_values = [10, 20] if args.quick else [10, 20, 30, 40]
    subtask_values = [25, 50] if args.quick else [25, 50, 75, 100]
    bw_values = [2, 8] if args.quick else [2, 4, 6, 8]
    speed_values = [10, 40] if args.quick else [10, 20, 30, 40]
    node_values = [25, 50] if args.quick else [25, 50, 75, 100]

    sweep_and_plot("num_tasks", task_values, fixed, adprl_lo, adprl_ee, args.out_dir, args.seed,
                   eval_episodes=args.eval_episodes, dqn_warmup=args.dqn_warmup, skip_dqn=args.skip_dqn)
    sweep_and_plot(
        "num_subtasks",
        [(s, s) for s in subtask_values],
        fixed, adprl_lo, adprl_ee, args.out_dir, args.seed,
        x_values=subtask_values, cfg_key="num_subtasks_range",
        eval_episodes=args.eval_episodes, dqn_warmup=args.dqn_warmup, skip_dqn=args.skip_dqn,
    )
    sweep_and_plot("bw_mean", bw_values, fixed, adprl_lo, adprl_ee, args.out_dir, args.seed,
                   eval_episodes=args.eval_episodes, dqn_warmup=args.dqn_warmup, skip_dqn=args.skip_dqn)
    sweep_and_plot("speed_mean", speed_values, fixed, adprl_lo, adprl_ee, args.out_dir, args.seed,
                   eval_episodes=args.eval_episodes, dqn_warmup=args.dqn_warmup, skip_dqn=args.skip_dqn)
    sweep_and_plot("num_nodes", node_values, fixed, adprl_lo, adprl_ee, args.out_dir, args.seed,
                   eval_episodes=args.eval_episodes, dqn_warmup=args.dqn_warmup, skip_dqn=args.skip_dqn)

    print("[eval] done.")


if __name__ == "__main__":
    main()
