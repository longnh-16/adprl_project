"""
ddpg_agent.py
-------------
Implements Algorithm 1 (ADPRL) from the paper:

    1. actor/critic + target actor/critic networks
    2. replay buffer B
    3. asynchronous data collection from `num_workers` parallel environments
       (simulating the "1 server / 4 workers" asynchronous setup mentioned in
       Sec. 6.1)
    4. soft target updates with polyak factor `tau`

Hyperparameters follow the values reported in the paper (Sec. 6.1):
    replay memory size   = 10000
    mini-batch size      = 64
    actor learning rate  = 0.001
    critic learning rate = 0.002
    reward decay (gamma) = 0.001   <-- as literally stated in the paper.
                                        NOTE: this is unusually low for a
                                        discount factor; we expose it as a
                                        configurable hyperparameter
                                        (`gamma`) so you can also try more
                                        conventional values (e.g. 0.99) if
                                        you find the very small gamma hurts
                                        long-horizon credit assignment for
                                        your workload sizes.
    workers (servers=1, workers=4)
"""

from __future__ import annotations
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from networks import Actor, Critic

Transition = namedtuple("Transition", "state action reward next_state done")


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class ADPRLAgent:
    """Asynchronous Deep Progressive Reinforcement Learning agent (DDPG core)."""

    def __init__(
        self,
        state_dim: int,
        num_nodes: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 2e-3,
        gamma: float = 0.001,
        tau: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_nodes = num_nodes
        self.action_dim = num_nodes + 2
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_dim, num_nodes).to(self.device)
        self.critic = Critic(state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(state_dim, num_nodes).to(self.device)
        self.critic_target = Critic(state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.exploration_sigma = 0.3   # Ornstein-Uhlenbeck-like Gaussian exploration

    # ------------------------------------------------------------------ #
    def act(self, state: np.ndarray, explore: bool = True):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).squeeze(0)
        if explore:
            noise = torch.randn_like(a) * self.exploration_sigma
            a = torch.clamp(a + noise, 0.0, 1.0)
            a[: self.num_nodes] = F.softmax(a[: self.num_nodes], dim=-1)
        return a.cpu()

    def decode_action(self, action_tensor):
        return Actor.decode(action_tensor, self.num_nodes)

    def store(self, state, action_tensor, reward, next_state, done):
        self.buffer.push(
            np.asarray(state, dtype=np.float32),
            action_tensor.numpy().astype(np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )

    # ------------------------------------------------------------------ #
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        state = torch.as_tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.stack(batch.action), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.as_tensor(
            np.stack(batch.next_state), dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # ---- critic update (Eq. 29-30 / Algo.1 lines 13-14) -------------
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            y = reward + self.gamma * (1 - done) * target_q
        q = self.critic(state, action)
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor update (Algo.1 line 15, deterministic policy grad) ---
        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- soft target updates (Algo.1 lines 16-17) --------------------
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor"])
        self.critic_target.load_state_dict(ckpt["critic"])
