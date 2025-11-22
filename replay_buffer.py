
from collections import deque
import random
import numpy as np

class NStepReplayBuffer:
    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.n_step = n_step
        self.gamma = gamma

    def __len__(self):
        return len(self.buffer)

    def push_n_step(self, n_step_queue):
        """Take a deque of up to n transitions and collapse into a single n-step transition.

        Each element: (state, action, reward, next_state, done)
        """
        if len(n_step_queue) == 0:
            return

        state0, action0, _, _, _ = n_step_queue[0]

        R = 0.0
        gamma_pow = 1.0
        done_n = False
        next_state_n = n_step_queue[-1][3]

        for (s, a, r, ns, d) in n_step_queue:
            R += gamma_pow * r
            gamma_pow *= self.gamma
            next_state_n = ns
            if d:
                done_n = True
                break

        gamma_n = gamma_pow

        transition = (state0, action0, R, next_state_n, done_n, gamma_n)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, gammas = zip(*batch)
        return (
            np.array(states, dtype=float),
            np.array(actions, dtype=int),
            np.array(rewards, dtype=float),
            np.array(next_states, dtype=float),
            np.array(dones, dtype=bool),
            np.array(gammas, dtype=float),
        )
