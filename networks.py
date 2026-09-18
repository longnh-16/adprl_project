"""
networks.py
-----------
Actor and Critic networks for the DDPG backbone of ADPRL (Sec. 5.5,
Fig. 4). The actor outputs:
    - a categorical distribution (as logits) over which edge node to
      offload the current subtask to  (delta_{i,j,n,t} in the paper)
    - a bandwidth fraction in [0,1]  (eta_{f,i,j}^t, scaled by the caller
      to the actual available bandwidth)
    - a voluntary-wait fraction in [0,1]  (epsilon_{i,j}^t)

We use the standard "Gumbel-Softmax-free" trick for a DDPG-compatible
discrete-action head: the actor outputs continuous logits for the node
choice and the environment/agent takes argmax at execution time while
exploration noise is added directly to the logits (this keeps the network
fully differentiable for the DDPG policy gradient, matching Algorithm 1's
update rule, while still producing a discrete offloading decision).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim: int, num_nodes: int, hidden: int = 128):
        super().__init__()
        self.num_nodes = num_nodes
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.node_head = nn.Linear(hidden, num_nodes)     # logits -> softmax
        self.bw_head = nn.Linear(hidden, 1)                # -> sigmoid, [0,1]
        self.wait_head = nn.Linear(hidden, 1)               # -> sigmoid, [0,1]

    def forward(self, state):
        h = self.backbone(state)
        node_logits = self.node_head(h)
        node_probs = F.softmax(node_logits, dim=-1)
        bw = torch.sigmoid(self.bw_head(h))
        wait = torch.sigmoid(self.wait_head(h))
        # action vector fed to the critic: concat(node_probs, bw, wait)
        action = torch.cat([node_probs, bw, wait], dim=-1)
        return action

    @staticmethod
    def decode(action_tensor, num_nodes):
        """Turns the actor's continuous output into an env-ready action dict."""
        node_probs = action_tensor[:num_nodes]
        bw = float(action_tensor[num_nodes])
        wait = float(action_tensor[num_nodes + 1])
        node = int(torch.argmax(node_probs).item())
        return {"node": node, "bandwidth": bw, "wait": wait}


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
