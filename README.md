
# RL Snake v5 – 16-Environment 4x4 Grid Visualization 🐍🐍🐍🐍

This version focuses on **visualizing 16 Snake environments at once** in a single Pygame window,
while training a **Dueling Double DQN + N-step** agent.

## Features

- 4 × 4 grid of Snake games (16 environments).
- All envs run in parallel and are displayed live.
- Dueling DQN (value + advantage head).
- Double DQN with a target network.
- N-step returns via an n-step replay buffer.
- Epsilon-decay exploration.
- Logging to Weights & Biases (wandb).
- Optional live matplotlib plot of score & mean score.

## Files

- `game.py` — `SnakeGame` environment that:
  - Manages its own `pygame.Surface`, not the main window.
  - Exposes `step()`, `get_state()`, `render()`, and `get_surface()`.

- `model.py` — `DuelingQNet` + `QTrainer` (Double DQN with target network).

- `replay_buffer.py` — `NStepReplayBuffer` that stores:
  `(state, action_idx, R_n, next_state_n, done_n, gamma_n)`.

- `agent.py` — Agent with:
  - Dueling network.
  - n-step replay.
  - Epsilon-greedy exploration.
  - Automatic load of saved model if present.

- `train_v5.py` — Multi-environment training loop:
  - Creates 16 envs in a 4×4 grid.
  - Single Pygame window of size 1280 × 960 (16 × (320 × 240)).
  - Trains the agent from all environments in parallel.
  - Logs to wandb project `rl_snake_v5_multi_env_grid`.

## Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Weights & Biases (wandb)

Login once:

```bash
wandb login
```

Paste your API key from https://wandb.ai

## Run Training

```bash
python train_v5.py
```

You will see:

- A 4 × 4 grid of Snake games, all learning live.
- Terminal stats per episode (env index, score, best, mean, epsilon).
- Wandb logs of episode rewards and training metrics.

You can change hyperparameters like:

- Number of envs (currently fixed at 4×4 = 16).
- N-step length.
- Learning rate, batch size.

inside `train_v5.py` in the `config` dictionary.
