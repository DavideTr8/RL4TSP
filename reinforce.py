from mlp_agent import MLPAgent
from gym_env import TSPEnv
import torch

from utils import greedy_policy
from tqdm import tqdm

class REINFORCE:
    def __init__(self, n_nodes):
        self.env = TSPEnv(n_nodes)
        self.agent = MLPAgent(n_nodes)
        self.buffer = []
        self.scores = []

        # Hyperparameters
        self.training_epochs = 50
        self.episodes_per_epoch = 1000
        self.gamma = 1.0
        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)


    def collect_one_episode(self):
        state = self.env.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            action_probs = self.agent(state)
            action = torch.multinomial(action_probs, 1).item()
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            log_probs.append(torch.log(action_probs[:, action]))
        self.buffer.append((rewards, log_probs, state))

    def update(self):
        loss = 0
        for rewards, log_probs, state in self.buffer:
            # compute reward for a greedy policy
            greedy_policy_length, _ = greedy_policy(state["nodes"])

            returns = []
            R = 0
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)

            log_probs = torch.stack(log_probs)
            loss -= torch.sum(log_probs) * (R + greedy_policy_length)
            self.scores.append(R + greedy_policy_length)
        self.optimizer.zero_grad()
        (loss / len(self.buffer)).backward()
        self.optimizer.step()
        self.buffer = []
        

    def train(self):
        with tqdm(total=self.training_epochs, position=0, desc="Epoch") as pbar:
            for epoch_num in range(self.training_epochs):
                for _ in tqdm(range(self.episodes_per_epoch), desc="Episode", position=1, leave=False):
                    self.collect_one_episode()
                self.update()
                pbar.update(1)
                last_score = sum(self.scores[-self.episodes_per_epoch:])
                mean_score = last_score / self.episodes_per_epoch
                tqdm.write(f"Epoch {epoch_num + 1}, score: {mean_score:.2f}")


if __name__ == "__main__":
    reinforce = REINFORCE(n_nodes=10)
    reinforce.train()

    from matplotlib import pyplot as plt
    plt.plot(reinforce.scores)
    plt.savefig("reinforce.png")
    plt.show()