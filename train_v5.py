
import pygame
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import wandb

from game import SnakeGame
from agent import Agent

plt.ion()

GRID_ROWS = 4
GRID_COLS = 4
ENV_W = 320
ENV_H = 240
NUM_ENVS = GRID_ROWS * GRID_COLS
FPS = 45

def main():
    config = {
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "num_envs": NUM_ENVS,
        "gamma": 0.99,
        "n_step": 3,
        "batch_size": 512,
        "lr": 1e-4,
        "buffer_capacity": 200_000,
        "fps": FPS,
    }
    wandb.init(project="rl_snake_v5_multi_env_grid", config=config)

    pygame.init()
    screen = pygame.display.set_mode((GRID_COLS * ENV_W, GRID_ROWS * ENV_H))
    pygame.display.set_caption("RL Snake v5 - 4x4 Grid")
    clock = pygame.time.Clock()

    envs = [SnakeGame(ENV_W, ENV_H) for _ in range(NUM_ENVS)]
    states = [env.get_state() for env in envs]
    n_step_queues = [deque(maxlen=config["n_step"]) for _ in range(NUM_ENVS)]

    agent = Agent(
        state_dim=len(states[0]),
        num_actions=3,
        gamma=config["gamma"],
        n_step=config["n_step"],
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
        plt.title("Training Progress (v5)")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(scores, label="Score")
        plt.plot(mean_scores, label="Mean Score")
        plt.legend()
        plt.pause(0.1)

    running = True
    while running:
        # handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # step all environments once
        for idx, env in enumerate(envs):
            state = states[idx]
            action_idx = agent.select_action(state)

            # one-hot for env
            action_vec = np.zeros(3, dtype=int)
            action_vec[action_idx] = 1

            reward, done, score = env.step(action_vec)
            next_state = env.get_state()
            agent.total_steps += 1

            # update n-step queue
            n_step_queues[idx].append((state, action_idx, reward, next_state, done))
            if len(n_step_queues[idx]) == config["n_step"] or done:
                agent.remember_n_step(n_step_queues[idx])
                n_step_queues[idx].clear()

            # train
            loss = agent.train(config["batch_size"])

            # episode end for this env
            if done:
                episodes += 1
                env.reset()
                states[idx] = env.get_state()

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
                    f"Ep {episodes} | Env {idx} | Score {score} | "
                    f"Best {best_score} | Mean {mean_score:.2f} | Eps {agent.epsilon():.3f}"
                )
                plot()
            else:
                states[idx] = next_state

        # render grid
        screen.fill((0, 0, 0))
        for idx, env in enumerate(envs):
            env.render()
            surf = env.get_surface()
            row = idx // GRID_COLS
            col = idx % GRID_COLS
            screen.blit(surf, (col * ENV_W, row * ENV_H))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
