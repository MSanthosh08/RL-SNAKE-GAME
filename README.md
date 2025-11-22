
# Reinforcement Learning Snake (DQN)

This project is a Snake game where the snake learns to play using Deep Q-Learning (DQN).

## Features

- Custom Snake environment built with Pygame
- Deep Q-Network (PyTorch) for decision making
- Experience replay and epsilon-greedy exploration
- Live visualization of the Snake during training

## Project Structure

- `game.py` — Snake environment (state, rewards, rendering)
- `model.py` — Neural network (Q-network) and trainer
- `agent.py` — DQN agent with replay memory and action selection
- `train.py` — Training loop (runs the game + learning)
- `requirements.txt` — Python dependencies

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training

```bash
python train.py
```

By default, the game window is visible while training.  
If you want faster training without visualization, edit:

```python
agent = Agent(render=False)
```

in `train.py`.

## How It Works (High-Level)

- The agent observes the current state of the game (danger, direction, food location).
- It chooses an action: go straight, turn right, or turn left.
- The environment returns a reward and the next state.
- Using DQN, the agent updates its neural network to improve future decisions.
