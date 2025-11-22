
import numpy as np
import torch
import random

from model import DuelingQNet, QTrainer
from replay_buffer import NStepReplayBuffer

class Agent:
    def __init__(self, state_dim, num_actions=3, gamma=0.99, n_step=3,
                 buffer_capacity=100_000, lr=1e-4):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step

        self.model = DuelingQNet(state_dim, 256, num_actions)
        self.target_model = DuelingQNet(state_dim, 256, num_actions)
        self.trainer = QTrainer(self.model, self.target_model, lr=lr)
        self.memory = NStepReplayBuffer(buffer_capacity, n_step, gamma)

        self.n_games = 0
        self.total_steps = 0
        self.epsilon_start = 1.0
        self.epsilon_final = 0.05
        self.epsilon_decay = 200_000  # steps

        # Try loading existing model (resume)
        self.model.load()
        self.target_model.load()

    def epsilon(self):
        # linear decay
        return max(
            self.epsilon_final,
            self.epsilon_start - (self.total_steps / self.epsilon_decay),
        )

    def get_action(self, state):
        """Return discrete action index (0,1,2)."""
        eps = self.epsilon()
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)

        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        action_idx = torch.argmax(q_values, dim=1).item()
        return int(action_idx)

    def remember_n_step(self, n_step_queue, gamma):
        """Collapse n-step queue into single transition and push to buffer."""
        self.memory.push_n_step(n_step_queue)

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones, gammas = self.memory.sample(batch_size)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones, gammas)
        return loss
