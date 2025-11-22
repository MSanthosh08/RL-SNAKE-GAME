
from agent import Agent
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.legend(['Score', 'Mean Score'])
    plt.pause(0.1)

def train():
    agent = Agent(render=True)
    scores = []
    mean_scores = []
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
            mean_scores.append(sum(scores) / len(scores))
            print(f"Game {agent.n_games}  Score {score}  Record {record}")

            plot(scores, mean_scores)

if __name__ == "__main__":
    train()
