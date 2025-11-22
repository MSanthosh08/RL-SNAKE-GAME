
import numpy as np
import torch
import random

from model import DuelingQNet, QTrainer
from replay_buffer import NStepReplayBuffer

class Agent:
    def __init__(self, state_dim, num_actions=3, gamma=0.99, n_step=3,
                 buffer_capacity=200_000, lr=1e-4):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step

        self.model = DuelingQNet(state_dim, 256, num_actions)
        self.target_model = DuelingQNet(state_dim, 256, num_actions)
        self.trainer = QTrainer(self.model, self.target_model, lr=lr)
        self.memory = NStepReplayBuffer(buffer_capacity, n_step, gamma)

        self.n_episodes = 0
        self.total_steps = 0

        self.eps_start = 1.0
        self.eps_final = 0.05
        self.eps_decay = 250_000  # steps

        # load if exists
        self.model.load()
        self.target_model.load()

    def epsilon(self):
        return max(
            self.eps_final,
            self.eps_start - self.total_steps / self.eps_decay,
        )

    def select_action(self, state):
        eps = self.epsilon()
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(state_t)
        return int(torch.argmax(q_vals, dim=1).item())

    def remember_n_step(self, n_step_queue):
        self.memory.push_n_step(n_step_queue)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        states, actions, rewards, next_states, dones, gammas = self.memory.sample(batch_size)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones, gammas)
        return loss
