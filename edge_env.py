"""
edge_env.py
-----------
A simulation of the Collaborative Edge Computing (CEC) system described in
Section 4 of the paper ("System Model and Problem Formulation").

This environment implements:
    - a random network topology of edge nodes/links (Sec. 6.1 "Network Model")
    - the per-subtask computation-time model of Eq. (1)
    - the flow-communication-time model of Eq. (3)-(5)
    - the energy-consumption model of Eq. (6)-(10)
    - the QoS / reward model of Eq. (11) and (23)

It exposes a Gym-like API:

    state = env.reset(tasks)
    for step in range(num_subtasks_total):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

`action` is a dict with:
    node        : int,   index of the edge node chosen for the current subtask
    bandwidth   : float, fraction in [0,1] of the *maximum available* bandwidth
                  on the path used for this subtask's incoming flow
    wait        : float, fraction in [0,1] representing extra voluntary wait
                  (epsilon_f in the paper) -- included for completeness, the
                  default agents mostly leave this at 0.

The environment schedules subtasks of all tasks in *dependency + release-time*
order (an ODTO online scheduler processes subtasks as they become ready).
"""

from __future__ import annotations
import numpy as np
import networkx as nx
import random
from dataclasses import dataclass, field


@dataclass
class EdgeNode:
    node_id: int
    proc_speed: float       # Mcps (megacycles/sec)
    comp_power: float       # Watts, used for computation energy Eq.(6)


@dataclass
class EdgeLink:
    u: int
    v: int
    bandwidth: float        # Mbps, total capacity
    tx_power: float         # Watts, used for transmission energy Eq.(7)


class NetworkModel:
    """Random connected topology of edge nodes with heterogeneous processing
    speed and link bandwidth (Sec. 6.1)."""

    def __init__(
        self,
        num_nodes: int,
        proc_speed_mean: float = 40.0,
        proc_speed_cv: float = 0.8,
        bandwidth_mean: float = 10.0,
        bandwidth_cv: float = 0.8,
        link_prob: float = 0.35,
        seed: int | None = None,
    ):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)

        self.nodes = []
        for i in range(num_nodes):
            speed = max(1.0, self.rng.normal(proc_speed_mean, proc_speed_cv * proc_speed_mean))
            power = self.rng.uniform(2.0, 6.0)  # Watts, computation power P_n^c
            self.nodes.append(EdgeNode(i, speed, power))

        # random connected topology (Erdos-Renyi, then patched to be connected)
        g = nx.erdos_renyi_graph(num_nodes, link_prob, seed=seed)
        if not nx.is_connected(g):
            comps = list(nx.connected_components(g))
            for i in range(len(comps) - 1):
                u = self.pyrng.choice(list(comps[i]))
                v = self.pyrng.choice(list(comps[i + 1]))
                g.add_edge(u, v)
        self.graph = g

        self.links = {}
        for (u, v) in g.edges:
            bw = max(0.5, self.rng.normal(bandwidth_mean, bandwidth_cv * bandwidth_mean))
            tx_power = self.rng.uniform(0.5, 2.0)  # Watts, P_u^x
            self.links[(u, v)] = EdgeLink(u, v, bw, tx_power)
            self.links[(v, u)] = EdgeLink(v, u, bw, tx_power)

        # remaining bandwidth per (directed) link at "current" time, refreshed per episode
        self.reset_bandwidth()

    def reset_bandwidth(self):
        self.remaining_bw = {k: l.bandwidth for k, l in self.links.items()}
        self.node_busy_until = {n.node_id: 0.0 for n in self.nodes}  # for waiting time proxy

    def path_bandwidth(self, src: int, dst: int):
        """Returns (path, min_remaining_bandwidth) along the shortest path."""
        if src == dst:
            return [src], float("inf")
        try:
            path = nx.shortest_path(self.graph, src, dst)
        except nx.NetworkXNoPath:
            return None, 0.0
        min_bw = min(
            self.remaining_bw.get((path[i], path[i + 1]), 0.0)
            for i in range(len(path) - 1)
        )
        return path, min_bw

    def consume_bandwidth(self, path, amount, duration):
        """Consume `amount` Mbps for `duration` seconds on every link of the path
        (very simplified capacity accounting -- released after `duration`)."""
        for i in range(len(path) - 1):
            k = (path[i], path[i + 1])
            if k in self.remaining_bw:
                self.remaining_bw[k] = max(0.0, self.remaining_bw[k] - amount)


class CECEnv:
    """Online dependent-task offloading environment (ODTO), Sec. 4-5."""

    def __init__(self, network: NetworkModel, lambda_t: float = 0.5, lambda_e: float = 0.5):
        self.net = network
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e
        self.PTx = 0.6   # transmission power (uplink), Watts
        self.PRx = 0.3   # reception power (downlink), Watts
        self.reset_stats()

    # ------------------------------------------------------------------ #
    def reset_stats(self):
        self._TT_max = 1.0
        self._EC_max = 1.0

    def reset(self, tasks):
        """Prepares the schedule queue for a new episode/workload."""
        self.net.reset_bandwidth()
        self.tasks = tasks
        self.finish_time = {}            # (task_id, subtask) -> finish time
        self.task_energy = {t.task_id: 0.0 for t in tasks}
        self.task_finish = {}
        self.clock = 0.0
        self.offloading_ratio_count = [0, 0]  # [offloaded, total]

        # build a global ready queue respecting DAG + release-time order
        self._build_queue()
        return self._observe()

    def _build_queue(self):
        """Flattens all subtasks into an execution queue honoring topological
        order per task and release time across tasks (Earliest-Release-first
        tie-break, matching the ERTF heuristic used as one of the paper's
        baselines for ordering ready subtasks)."""
        self.queue = []
        for t in self.tasks:
            for j in t.topo_order():
                self.queue.append((t.task_id, j))
        # stable sort by release time (keeps topo order within a task)
        release_of = {t.task_id: t.release_time for t in self.tasks}
        self.queue.sort(key=lambda x: release_of[x[0]])
        self.ptr = 0
        self.by_id = {t.task_id: t for t in self.tasks}

    # ------------------------------------------------------------------ #
    def _current_subtask(self):
        if self.ptr >= len(self.queue):
            return None
        return self.queue[self.ptr]

    def _observe(self):
        cur = self._current_subtask()
        if cur is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        task_id, j = cur
        task = self.by_id[task_id]
        data = task.graph.nodes[j]["data"]
        comp = task.graph.nodes[j]["comp"]
        preds = task.predecessors(j)
        n_preds = len(preds)
        pred_finish = max([self.finish_time.get((task_id, p), 0.0) for p in preds], default=0.0)

        # network summary features: average remaining bandwidth ratio, per-node load
        bw_ratios = [
            self.net.remaining_bw[k] / max(self.net.links[k].bandwidth, 1e-6)
            for k in self.net.remaining_bw
        ]
        avg_bw_ratio = float(np.mean(bw_ratios)) if bw_ratios else 1.0
        node_load = [
            max(0.0, self.net.node_busy_until[n.node_id] - self.clock) for n in self.net.nodes
        ]
        avg_node_wait = float(np.mean(node_load)) if node_load else 0.0
        max_node_wait = float(np.max(node_load)) if node_load else 0.0

        feats = [
            data / 500.0,  # normalized data size
            comp / 500.0,
            n_preds / 5.0,
            pred_finish / 100.0,
            task.release_time / 20.0,
            avg_bw_ratio,
            avg_node_wait / 50.0,
            max_node_wait / 50.0,
            self.clock / 100.0,
            (len(self.queue) - self.ptr) / max(len(self.queue), 1),
        ]
        return np.array(feats, dtype=np.float32)

    @property
    def state_dim(self):
        return 10

    @property
    def action_dim(self):
        # [node_choice(continuous, decoded via argmax over num_nodes logits),
        #  bandwidth fraction, wait fraction]
        return self.net.num_nodes + 2

    # ------------------------------------------------------------------ #
    def step(self, action: dict):
        """`action`:
            node: int chosen edge-node id
            bandwidth: float in [0,1], fraction of the max available path bw
            wait: float in [0,1], voluntary extra wait (as a fraction of 5s)
        """
        cur = self._current_subtask()
        if cur is None:
            return self._observe(), 0.0, True, {}

        task_id, j = cur
        task = self.by_id[task_id]
        node_id = int(np.clip(action["node"], 0, self.net.num_nodes - 1))
        bw_frac = float(np.clip(action["bandwidth"], 0.0, 1.0))
        wait_frac = float(np.clip(action.get("wait", 0.0), 0.0, 1.0))

        data = task.graph.nodes[j]["data"]
        comp = task.graph.nodes[j]["comp"]
        preds = task.predecessors(j)
        pred_finish = max([self.finish_time.get((task_id, p), 0.0) for p in preds], default=0.0)

        src = task.source_node
        offloaded = int(node_id != src)
        self.offloading_ratio_count[1] += 1
        self.offloading_ratio_count[0] += offloaded

        # ---- flow communication time (Eq. 3-4) -------------------------
        path, max_bw = self.net.path_bandwidth(src, node_id)
        extra_wait = wait_frac * 5.0
        if path is None or max_bw <= 0:
            bw_alloc = 0.1
            comm_time = data / bw_alloc + extra_wait
        else:
            bw_alloc = max(0.1, bw_frac * max_bw)
            comm_time = data / bw_alloc + extra_wait
            self.net.consume_bandwidth(path, bw_alloc, comm_time)

        start_time = max(self.clock, pred_finish, task.graph.nodes[j]["release"])
        # ---- computation time (Eq. 1) ----------------------------------
        node = self.net.nodes[node_id]
        node_wait = max(0.0, self.net.node_busy_until[node_id] - start_time)
        comp_time = comp / node.proc_speed + node_wait

        finish = start_time + comm_time + comp_time
        self.finish_time[(task_id, j)] = finish
        self.net.node_busy_until[node_id] = max(self.net.node_busy_until[node_id], finish)
        self.clock = max(self.clock, start_time)

        # ---- energy consumption (Eq. 6-8) -------------------------------
        ec_comp = node.comp_power * comp_time
        link_tx_power = self.net.links.get((src, node_id), None)
        px = link_tx_power.tx_power if link_tx_power else 1.0
        ec_tx = px * comm_time
        ec_subtask = ec_comp + ec_tx
        self.task_energy[task_id] += ec_subtask

        self.ptr += 1
        done = self.ptr >= len(self.queue)
        if done:
            for t in self.tasks:
                subtask_finishes = [
                    self.finish_time.get((t.task_id, k), 0.0) for k in t.graph.nodes
                ]
                self.task_finish[t.task_id] = max(subtask_finishes) if subtask_finishes else 0.0
            self._finalize_reward()

        reward = self._step_reward(comp_time, comm_time, ec_subtask)
        info = {"finish": finish, "energy": ec_subtask, "offloaded": offloaded}
        return self._observe(), reward, done, info

    # ------------------------------------------------------------------ #
    def _step_reward(self, comp_time, comm_time, energy):
        """Dense, dependency-aware shaping reward used *during* an episode
        (a normalized-negative combination, consistent in spirit with the
        progressive reward described in Sec. 5.4, Eq. 23, but computed
        per-subtask so that the DDPG agent gets a training signal at every
        step instead of only at episode end)."""
        tt_norm = (comp_time + comm_time) / 50.0
        ec_norm = energy / 20.0
        return -(self.lambda_t * tt_norm + self.lambda_e * ec_norm)

    def _finalize_reward(self):
        if not self.task_finish:
            return
        TT = float(np.mean(list(self.task_finish.values())))
        EC = float(np.sum(list(self.task_energy.values())))
        self._TT_max = max(self._TT_max, TT)
        self._EC_max = max(self._EC_max, EC)
        self.last_ACT = TT
        self.last_EC = EC
        self.last_OR = (
            self.offloading_ratio_count[0] / max(self.offloading_ratio_count[1], 1)
        )

    def episode_metrics(self):
        """Returns (ACT, EC, OR) for the just-finished episode."""
        return getattr(self, "last_ACT", 0.0), getattr(self, "last_EC", 0.0), getattr(
            self, "last_OR", 0.0
        )
