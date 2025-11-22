
from agent import Agent

def train():
    agent = Agent(render=True)  # set render=False for faster training without visualization
    scores = []
    mean_scores = []
    record = 0

    while True:
        # 1. Get old state
        state_old = agent.get_state()

        # 2. Get move
        final_move = agent.get_action(state_old)

        # 3. Perform move and get new state
        reward, done, score = agent.game.play_step(final_move)
        state_new = agent.get_state()

        # 4. Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            agent.game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            scores.append(score)
            mean_score = sum(scores) / len(scores)
            mean_scores.append(mean_score)

            print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Mean: {mean_score:.2f}")

if __name__ == "__main__":
    train()
