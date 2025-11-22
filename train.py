
from agent import Agent
import matplotlib.pyplot as plt
from IPython import display
import csv
import os

plt.ion()

LOG_FILE = "training_log.csv"

def plot(scores, mean_scores, records):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(records)
    plt.legend(["Score", "Mean Score", "Record"])
    plt.pause(0.1)

def append_log(game, score, mean_score, record):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["game", "score", "mean_score", "record"])
        writer.writerow([game, score, mean_score, record])

def train():
    agent = Agent(render=True)
    scores = []
    mean_scores = []
    records = []
    record = 0

    while True:
        state_old = agent.get_state()
        final_move = agent.get_action(state_old)
        reward, done, score = agent.game.play_step(final_move)
        state_new = agent.get_state()

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            agent.game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            scores.append(score)
            mean_score = sum(scores) / len(scores)
            mean_scores.append(mean_score)
            records.append(record)

            print(f"Game {agent.n_games}  Score {score}  Record {record}  Mean {mean_score:.2f}")

            append_log(agent.n_games, score, mean_score, record)
            plot(scores, mean_scores, records)

if __name__ == "__main__":
    train()
