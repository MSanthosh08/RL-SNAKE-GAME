
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="model.pth"):
        model_folder_path = "./model"
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name, map_location=torch.device("cpu")))
            print(f"Loaded model from {file_name}")
        else:
            print("No saved model found, starting fresh.")


class QTrainer:
    def __init__(self, model, target_model, lr, gamma, tau=0.005, update_every=100):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction="none")  # we'll apply PER weights
        self.update_every = update_every
        self.step_count = 0
        # Initialize target model
        self.target_model.load_state_dict(self.model.state_dict())
        self.tau = tau  # for soft update if desired

    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done, weights=None):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
            if weights is not None:
                weights = torch.tensor(weights, dtype=torch.float).unsqueeze(0)

        # Online network estimates
        q_values = self.model(state)  # (batch, num_actions)
        # Chosen actions
        if action.dim() == 1:
            actions_idx = torch.argmax(action).unsqueeze(0).unsqueeze(1)  # indices of actions taken
        else:
            actions_idx = torch.argmax(action, dim=1).unsqueeze(1)
        q_value = q_values.gather(1, actions_idx).squeeze(1)

        # Double DQN: actions from online net, values from target net
        with torch.no_grad():
            next_q_values_online = self.model(next_state)
            next_actions = torch.argmax(next_q_values_online, dim=1, keepdim=True)  # (batch,1)
            next_q_values_target = self.target_model(next_state)
            next_q_value = next_q_values_target.gather(1, next_actions).squeeze(1)
            q_target = reward + self.gamma * next_q_value * (~done)

        td_error = q_target - q_value
        loss_elements = self.criterion(q_value, q_target)

        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float)
            if w.shape[0] != loss_elements.shape[0]:
                w = w[: loss_elements.shape[0]]
            loss = (loss_elements * w).mean()
        else:
            loss = loss_elements.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self.hard_update()

        return td_error.detach().cpu().numpy()
