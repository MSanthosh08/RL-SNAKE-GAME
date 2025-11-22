
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class DuelingQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_adv = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc_value(x)
        adv = self.fc_adv(x)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = value + adv - adv_mean
        return q

    def save(self, file_name="model_dueling.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

    def load(self, file_name="model_dueling.pth"):
        model_folder_path = "./model"
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
            print(f"Loaded model weights from {file_path}")
        else:
            print("No saved model found, starting with random weights.")


class QTrainer:
    def __init__(self, model, target_model, lr=1e-4):
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        # sync target
        self.target_model.load_state_dict(self.model.state_dict())
        self.step_count = 0
        self.target_update_freq = 1000  # steps

    def train_step(self, states, actions, rewards, next_states, dones, gammas):
        """Train with Double DQN + optional n-step gamma per sample.

        states: (B, state_dim)
        actions: (B,) int indices
        rewards: (B,)
        next_states: (B, state_dim)
        dones: (B,) bool
        gammas: (B,) discount factors (e.g., gamma**n for n-step)
        """
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)
        gammas = torch.tensor(gammas, dtype=torch.float)

        # current Q-values
        q_values = self.model(states)  # (B, num_actions)
        action_indices = actions.unsqueeze(1)
        q_value = q_values.gather(1, action_indices).squeeze(1)  # (B,)

        # Double DQN target: online selects action, target evaluates
        with torch.no_grad():
            next_q_online = self.model(next_states)         # (B, A)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)  # (B,1)
            next_q_target = self.target_model(next_states)  # (B, A)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)        # (B,)

            target = rewards + gammas * next_q * (~dones)

        loss = self.criterion(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()
