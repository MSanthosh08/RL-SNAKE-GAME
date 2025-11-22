
import matplotlib.pyplot as plt
from IPython import display
from collections import deque

import numpy as np
import wandb

from game import SnakeGameAI
from agent import Agent

plt.ion()

def main():
    config = {
        "num_envs": 4,
        "gamma": 0.99,
        "n_step": 3,
        "batch_size": 512,
        "lr": 1e-4,
        "buffer_capacity": 100_000,
    }

    wandb.init(project="rl_snake_v4", config=config)

    num_envs = config["num_envs"]
    gamma = config["gamma"]
    n_step = config["n_step"]
    batch_size = config["batch_size"]

    # First env renders, others offscreen (no window)
    envs = [
        SnakeGameAI(render=(i == 0))
        for i in range(num_envs)
    ]
    states = [env.get_state() for env in envs]
    n_step_queues = [deque(maxlen=n_step) for _ in range(num_envs)]

    agent = Agent(
        state_dim=len(states[0]),
        num_actions=3,
        gamma=gamma,
        n_step=n_step,
        buffer_capacity=config["buffer_capacity"],
        lr=config["lr"],
    )

    scores = []
    mean_scores = []
    best_score = 0
    episodes = 0

    fig = plt.gcf()

    def plot():
        display.clear_output(wait=True)
        display.display(fig)
        plt.clf()
        plt.title("Training Progress (v4)")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(scores, label="Score")
        plt.plot(mean_scores, label="Mean Score")
        plt.legend()
        plt.pause(0.1)

    while True:
        # Step all envs once
        for i, env in enumerate(envs):
            state = states[i]
            action_idx = agent.get_action(state)

            # convert discrete action to one-hot for env
            action_vec = np.zeros(3, dtype=int)
            action_vec[action_idx] = 1

            reward, done, score = env.play_step(action_vec)
            next_state = env.get_state()
            agent.total_steps += 1

            # add to n-step buffer for this env
            n_step_queues[i].append((state, action_idx, reward, next_state, done))

            # if we have n steps or episode ended, collapse and store
            if len(n_step_queues[i]) == n_step or done:
                agent.remember_n_step(n_step_queues[i], gamma)
                n_step_queues[i].clear()

            # train step
            loss = agent.train_step(batch_size)

            # if episode end, reset env
            if done:
                episodes += 1
                env.reset()
                states[i] = env.get_state()

                scores.append(score)
                mean_score = sum(scores) / len(scores)
                mean_scores.append(mean_score)
                best_score = max(best_score, score)

                wandb.log(
                    {
                        "episode": episodes,
                        "score": score,
                        "mean_score": mean_score,
                        "best_score": best_score,
                        "epsilon": agent.epsilon(),
                        "steps": agent.total_steps,
                        **({"loss": loss} if loss is not None else {}),
                    }
                )

                print(
                    f"Ep {episodes} | Env {i} | Score {score} | "
                    f"Best {best_score} | Mean {mean_score:.2f} | Eps {agent.epsilon():.3f}"
                )

                plot()
            else:
                states[i] = next_state


if __name__ == "__main__":
    main()
