
import random
import numpy as np
import torch

from model import QTrainer
from model_cnn import CNN_QNet
from game import SnakeGameAI
from agent import PrioritizedReplayBuffer

MAX_MEMORY = 50_000
BATCH_SIZE = 64
LR = 0.00025

class CNNAgent:
    def __init__(self, render=True):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.99
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
        self.model = CNN_QNet(1, 3)
        self.target_model = CNN_QNet(1, 3)
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma, update_every=500)
        self.game = SnakeGameAI(render=render)

        self.model.load()
        self.target_model.load()

    def get_state(self):
        frame = self.game.get_frame(size=(84, 84))  # (H, W)
        # Add channel and batch dimension later in trainer
        return frame

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE)
        # reshape states to NCHW
        states = states[:, None, :, :]  # (B,1,H,W)
        next_states = next_states[:, None, :, :]
        td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
        self.memory.update_priorities(indices, td_errors)

    def train_short_memory(self, state, action, reward, next_state, done):
        s = state[None, None, :, :]
        ns = next_state[None, None, :, :]
        self.trainer.train_step(s, action, reward, ns, done, weights=None)

    def get_action(self, state):
        self.epsilon = max(5, 100 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return np.array(final_move, dtype=int)
