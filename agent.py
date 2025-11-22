
import random
from collections import deque
import numpy as np
import torch

from model import Linear_QNet, QTrainer
from game import SnakeGameAI

MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.0005

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        self.frame += 1
        beta = self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames
        beta = min(1.0, beta)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, priorities, eps=1e-5):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = abs(prio) + eps


class Agent:
    def __init__(self, render=True):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.99
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.target_model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)
        self.game = SnakeGameAI(render=render)

        # Try to load existing weights (resume training)
        self.model.load()
        self.target_model.load()

    def get_state(self):
        return self.game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE)
        td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
        self.memory.update_priorities(indices, td_errors)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Short memory without PER weights
        self.trainer.train_step(state, action, reward, next_state, done, weights=None)

    def get_action(self, state):
        self.epsilon = max(5, 80 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return np.array(final_move, dtype=int)
