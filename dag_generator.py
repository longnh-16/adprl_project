"""
dag_generator.py
-----------------
Generates synthetic DAG tasks, following the description in Section 6.1
("Synthetic Dataset") of the paper:

    Chen et al., "Dynamic Task Offloading in Edge Computing based on
    Dependency-aware Reinforcement Learning", IEEE TCC 2024.

Each task CT_i is a DAG (T_i, P_i):
    - T_i : set of subtasks
    - P_i : set of weighted dependency edges between subtasks

Per-task/subtask attributes generated:
    - data size (Mbit)            ~ N(500, (0.8*500)^2), clipped to be positive
    - computation load (KCC)      ~ N(500, (0.6*500)^2), clipped to be positive
    - release time                ~ Poisson(lambda)
    - source (edge) device        random among the first `source_nodes` nodes

NOTE: The paper does not give the *exact* random-DAG algorithm (it only cites
[23] for a general multi-layer DAG generator). We implement a standard
layer-by-layer random DAG generator (a common approach in DAG-scheduling
literature) that reproduces the qualitative properties described in the
paper: a DAG with several layers, a controllable number of dependency edges
between consecutive layers, and per-subtask data/computation attributes.
"""

from __future__ import annotations
import random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: int
    graph: nx.DiGraph            # DAG of subtasks; node attrs: data, comp, release
    source_node: int
    release_time: float

    @property
    def num_subtasks(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def topo_order(self):
        return list(nx.topological_sort(self.graph))

    def predecessors(self, j):
        return list(self.graph.predecessors(j))


def _random_layered_dag(num_subtasks: int, edge_density: float, rng: random.Random) -> nx.DiGraph:
    """Builds a random layered DAG with `num_subtasks` nodes.

    Nodes are split into `layers` layers (layers ~ Uniform[2, max(2,num_subtasks//2)]).
    Edges only go from a node in layer L to a node in a later layer, with
    probability controlled by `edge_density` (fraction of possible forward
    edges that are realised) -- this mirrors the paper's statement that the
    "number of edges (dependencies) between two layers ... is selected from
    a uniform distribution [1,100]" (we translate that count into a density).
    """
    num_subtasks = max(1, num_subtasks)
    n_layers = max(2, min(num_subtasks, rng.randint(2, max(2, num_subtasks // 2 + 1))))
    raw_layers = [[] for _ in range(n_layers)]
    for node in range(num_subtasks):
        raw_layers[rng.randrange(n_layers)].append(node)
    # compact away any empty layers so every layer we work with is non-empty
    layers = [l for l in raw_layers if l]
    if len(layers) < 2:
        # degenerate case (all nodes landed in one layer): split it in two
        all_nodes = layers[0] if layers else list(range(num_subtasks))
        mid = max(1, len(all_nodes) // 2)
        layers = [all_nodes[:mid], all_nodes[mid:]] if all_nodes[mid:] else [all_nodes]
        if len(layers) < 2:
            layers.append([])
    n_layers = len(layers)

    g = nx.DiGraph()
    g.add_nodes_from(range(num_subtasks))

    for li in range(n_layers - 1):
        src_layer = layers[li]
        if not src_layer:
            continue
        for lj in range(li + 1, n_layers):
            dst_layer = layers[lj]
            for u in src_layer:
                for v in dst_layer:
                    if rng.random() < edge_density:
                        g.add_edge(u, v)
        # guarantee at least one forward edge per source-layer node
        later_nonempty = [l for l in layers[li + 1:] if l]
        if later_nonempty and not any(g.out_degree(u) for u in src_layer):
            nxt = later_nonempty[0]
            for u in src_layer:
                v = rng.choice(nxt)
                if u != v:
                    g.add_edge(u, v)

    # ensure DAG is weakly connected (attach isolated components to node 0)
    if not nx.is_weakly_connected(g):
        comps = list(nx.weakly_connected_components(g))
        root = min(comps[0])
        for comp in comps[1:]:
            v = min(comp)
            g.add_edge(root, v)
    return g


class DAGTaskGenerator:
    def __init__(
        self,
        num_edge_nodes: int,
        source_nodes: int | None = None,
        data_mean: float = 500.0,      # Mbit
        data_cv: float = 0.8,
        comp_mean: float = 500.0,      # KCC
        comp_cv: float = 0.6,
        release_lambda: float = 5.0,
        subtask_range=(2, 12),
        edge_density_range=(0.15, 0.6),
        seed: int | None = None,
    ):
        self.num_edge_nodes = num_edge_nodes
        self.source_nodes = source_nodes or num_edge_nodes
        self.data_mean = data_mean
        self.data_cv = data_cv
        self.comp_mean = comp_mean
        self.comp_cv = comp_cv
        self.release_lambda = release_lambda
        self.subtask_range = subtask_range
        self.edge_density_range = edge_density_range
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _sample_positive_normal(self, mean, cv):
        val = self.np_rng.normal(mean, cv * mean)
        return float(max(val, 0.05 * mean))

    def generate_task(self, task_id: int, num_subtasks: int | None = None) -> Task:
        if num_subtasks is None:
            num_subtasks = self.rng.randint(*self.subtask_range)
        density = self.rng.uniform(*self.edge_density_range)
        g = _random_layered_dag(num_subtasks, density, self.rng)

        release_time = float(self.np_rng.poisson(self.release_lambda))
        for n in g.nodes:
            g.nodes[n]["data"] = self._sample_positive_normal(self.data_mean, self.data_cv)
            g.nodes[n]["comp"] = self._sample_positive_normal(self.comp_mean, self.comp_cv)
            g.nodes[n]["release"] = release_time
        for u, v in g.edges:
            # dependency transfer volume: a fraction of upstream node's data
            g.edges[u, v]["weight"] = g.nodes[u]["data"] * self.rng.uniform(0.2, 1.0)

        source_node = self.rng.randrange(self.source_nodes)
        return Task(task_id=task_id, graph=g, source_node=source_node, release_time=release_time)

    def generate_batch(self, num_tasks: int) -> list[Task]:
        return [self.generate_task(i) for i in range(num_tasks)]
