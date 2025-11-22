
# RL Snake – Advanced DQN Edition 🐍🤖

This version of the Snake RL project adds:

- ✅ Double DQN with target network
- ✅ Prioritized Experience Replay (PER)
- ✅ MLP agent using state features
- ✅ CNN agent using pixel observations
- ✅ Save / load models to resume training
- ✅ Live matplotlib plots during training
- ✅ Streamlit dashboard for monitoring training

## Structure

- `game.py` — Snake environment with improved reward shaping and state, plus `get_frame()` for CNN.
- `model.py` — Linear Q-network + `QTrainer` with Double DQN and target network.
- `agent.py` — MLP-based DQN agent using state vector + PER.
- `model_cnn.py` — CNN Q-network for pixel input.
- `agent_cnn.py` — DQN agent using CNN + PER.
- `train.py` — Trains the MLP agent; logs to `training_log.csv`.
- `train_cnn.py` — Trains the CNN agent; logs to `training_log_cnn.csv`.
- `dashboard.py` — Streamlit dashboard to visualize training logs.
- `requirements.txt` — Dependencies.

## Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train (State-Based Agent)

```bash
python train.py
```

This will:
- Open a Pygame window (snake learning)
- Show a live matplotlib plot
- Log metrics into `training_log.csv`

## Train (CNN Pixel-Based Agent)

```bash
python train_cnn.py
```

This uses the `get_frame()` output as CNN input and logs to `training_log_cnn.csv`.

> Note: CNN training is heavier; consider smaller window sizes or fewer games for testing.

## Streamlit Dashboard

In another terminal:

```bash
streamlit run dashboard.py
```

Then open the URL shown in the terminal.  
You’ll see separate tabs for:
- State-based MLP agent
- CNN agent

Both read from the CSV logs created during training.

## Resume Training

- Models are automatically saved into the `./model` folder.
- On startup, both agents attempt to load the saved weights.
