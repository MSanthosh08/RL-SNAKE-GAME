
# RL Snake v4 – Dueling DQN + N-step + Multi-env + WandB 🐍⚡

This is an advanced version of the RL Snake project, focusing on **clean RL tricks**:

- ✅ Dueling DQN heads (value + advantage)
- ✅ Double DQN (online + target networks)
- ✅ N-step returns (e.g., 3-step return)
- ✅ Multi-environment training (vectorized style)
- ✅ Logging to Weights & Biases (wandb)
- ✅ Live Matplotlib plotting (local)

> Note: This v4 project focuses on the **state-based agent** with a dueling MLP.  
> If you want the **CNN pixel-based agent**, you can still use the older v3 project.

## Files

- `game.py` — Snake environment + reward shaping + rich state.
- `model.py` — `DuelingQNet` and `QTrainer` with Double DQN + target net.
- `replay_buffer.py` — N-step replay buffer that stores (s, a, R_n, s_n, done_n, gamma_n).
- `agent.py` — Agent with epsilon-greedy, dueling net, n-step memory.
- `train_v4.py` — Multi-environment training loop with wandb + live plotting.

## Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Weights & Biases Setup

1. Install wandb (already in `requirements.txt`).
2. Login once:

   ```bash
   wandb login
   ```

   Paste your API key from https://wandb.ai

## Run Training

```bash
python train_v4.py
```

- One Snake window (env 0) will render.
- Other envs run offscreen to speed up experience collection.
- Training logs go to:
  - Local Matplotlib window (score, mean score)
  - Your wandb project: `rl_snake_v4`

You can customize:

- Number of parallel environments (`num_envs`)
- N-step horizon (`n_step`)
- Learning rate, batch size

inside `train_v4.py` in the `config` dictionary.
