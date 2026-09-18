"""
baselines.py
------------
Implements the four baseline policies compared against ADPRL in Sec. 6.1:

    - Random     : offload to a random edge node, random bandwidth fraction
    - LE         : Local Execution -- always run on the source node (no
                   offloading, no network flow)
    - Greedy     : pick the node with the minimum *estimated* completion time
                   (upload + queue + compute), flows scheduled FCFS
    - DQN+FCFS   : a DQN agent (discrete action = which node) that ignores
                   task dependency in its state representation, paired with
                   First-Come-First-Served bandwidth allocation (i.e. it
                   always asks for 100% of currently available bandwidth,
                   reflecting the paper's description that it "does not
                   consider task dependency" and schedules flows/computation
                   separately). This mirrors Tang & Wong [7] as cited in the
                   paper.

All baselines share the same `CECEnv` step() interface (action dict with
`node`, `bandwidth`, `wait`), so they can be dropped into the same
evaluation loop as ADPRL.
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple


class RandomPolicy:
    name = "Random"

    def __init__(self, num_nodes: int, seed: int | None = None):
        self.num_nodes = num_nodes
        self.rng = random.Random(seed)

    def act(self, state, env):
        return {
            "node": self.rng.randrange(self.num_nodes),
            "bandwidth": self.rng.random(),
            "wait": 0.0,
        }

    def observe(self, *args, **kwargs):
        pass


class LocalExecutionPolicy:
    name = "LE"

    def act(self, state, env):
        task_id, j = env._current_subtask()
        src = env.by_id[task_id].source_node
        return {"node": src, "bandwidth": 1.0, "wait": 0.0}

    def observe(self, *args, **kwargs):
        pass


class GreedyPolicy:
    """Chooses the node minimizing estimated (upload + queue + compute) time,
    Sec 6.1 baseline description; bandwidth requests use First-Come-First-
    Served (i.e., request full available bandwidth on the path)."""

    name = "Greedy"

    def act(self, state, env):
        task_id, j = env._current_subtask()
        task = env.by_id[task_id]
        data = task.graph.nodes[j]["data"]
        comp = task.graph.nodes[j]["comp"]
        src = task.source_node

        best_node, best_time = src, float("inf")
        for node in env.net.nodes:
            path, max_bw = env.net.path_bandwidth(src, node.node_id)
            bw = max(max_bw, 0.1) if path else 0.1
            comm_time = data / bw if node.node_id != src else 0.0
            queue_wait = max(0.0, env.net.node_busy_until[node.node_id] - env.clock)
            comp_time = comp / node.proc_speed + queue_wait
            est = comm_time + comp_time
            if est < best_time:
                best_time, best_node = est, node.node_id

        return {"node": best_node, "bandwidth": 1.0, "wait": 0.0}

    def observe(self, *args, **kwargs):
        pass


# ----------------------------------------------------------------------- #
#  DQN + FCFS
# ----------------------------------------------------------------------- #
DQNTransition = namedtuple("DQNTransition", "state action reward next_state done")


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNFCFSPolicy:
    """A dueling-free double-DQN agent over discrete node choices, ignoring
    task-dependency features in its (reduced) state, paired with FCFS
    bandwidth allocation -- approximating Tang & Wong [7] as described in
    the paper (Sec. 3.2, 6.1)."""

    name = "DQN+FCFS"

    def __init__(
        self,
        state_dim: int,
        num_nodes: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        buffer_size: int = 10000,
        batch_size: int = 64,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_nodes = num_nodes
        # DQN ignores dependency-specific features (n_preds, pred_finish) -> zero them out
        self.mask = np.ones(state_dim, dtype=np.float32)
        if state_dim >= 4:
            self.mask[2] = 0.0  # n_preds
            self.mask[3] = 0.0  # pred_finish

        self.q = QNetwork(state_dim, num_nodes).to(self.device)
        self.q_target = QNetwork(state_dim, num_nodes).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def _masked(self, state):
        return state * self.mask

    def act(self, state, env):
        s = self._masked(state)
        if random.random() < self.eps:
            node = random.randrange(self.num_nodes)
        else:
            with torch.no_grad():
                st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                node = int(torch.argmax(self.q(st), dim=-1).item())
        # FCFS: always request full available bandwidth
        return {"node": node, "bandwidth": 1.0, "wait": 0.0, "_discrete_action": node}

    def observe(self, state, action, reward, next_state, done):
        node = action.get("_discrete_action", action["node"])
        self.buffer.append(
            DQNTransition(self._masked(state), node, reward, self._masked(next_state), float(done))
        )
        self._learn()
        if done:
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def _learn(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        b = DQNTransition(*zip(*batch))
        state = torch.as_tensor(np.stack(b.state), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(b.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.as_tensor(b.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.as_tensor(np.stack(b.next_state), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(b.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(state).gather(1, action)
        with torch.no_grad():
            # double DQN target
            next_online_action = torch.argmax(self.q(next_state), dim=-1, keepdim=True)
            next_q = self.q_target(next_state).gather(1, next_online_action)
            y = reward + self.gamma * (1 - done) * next_q
        loss = F.mse_loss(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # periodic hard update of target net
        if random.random() < 0.01:
            self.q_target.load_state_dict(self.q.state_dict())
