import pygame
from mlp_agent import MLPAgent
from gym_env import TSPEnv
import torch

from utils import greedy_policy, random_policy
from tqdm import tqdm

class REINFORCE:
    """REINFORCE implementation that takes the Agent and the Environment, let them interact in episodes,
    saves episodes data in a buffer and uses such data to perform the training phase.
    As an additional information for monitoring the training, we also save the scores, where the score is the 
    difference between the Agent reward and the reward obtained by solving the same instance with a greedy policy.
    """
    def __init__(self, n_nodes):
        """The method for initialization creates the TSP environment and the MLP Agent.
        It then sets up the buffer and the scores attributes.
        Also, some hyperparameter for the algorithm are set.
        Finally, the SGD optimizer for the parameters update is created. We use Adam: https://arxiv.org/abs/1412.6980
        """
        self.env = TSPEnv(n_nodes)
        self.agent = MLPAgent(n_nodes, hidden_dim=256)
        self.buffer = []
        self.scores = []

        # Other hyperparameters
        self.training_epochs = 8192  # 1024
        self.episodes_per_epoch = 516  # 2048
        self.gamma = 1.0
        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)


    def collect_one_episode(self):
        """This is the method used for the data collection phase, that is the phase in which the Agens interacts
        with the Environment to populate the buffer.
        
        In this case, since in the REINFORCE algorithm we need the logarithms of the probabilities of the selected
        action instead of the action itself (see the loss function definition),
        to avoid computing them twice we save them now in the trajectory.
        Hence trajectories are composed of triplets (s_t, log(pi(a_t | s_t)), r_t).
        """
        state = self.env.reset()  # `reset` generates the starting state (a new instance for the TSP)
        done = False
        log_probs = []
        rewards = []
        while not done:
            action_probs = self.agent(state)
            action = torch.multinomial(action_probs, 1).item()
            state, reward, done, _ = self.env.step(action)  # `step` updates the state according to the action selected
            rewards.append(reward)
            log_probs.append(torch.log(action_probs[:, action]))
        self.buffer.append((rewards, log_probs, state))

    def update(self):
        """The update method is responsible of using the data present in the buffer to update the DNN
        parameters.
        
        Since instances are generated randomly, it might happen that the optimal solution for one instance might
        be longer than any solution for another instance. Hence, the length of a tour is not a good indicator
        for how well our agent is performing.
        For this reason, in the REINFORCE algorithm it is common to have a baseline used to evaluate the quality of the
        agent solution. In this case we use the solution computed by a greedy policy.
        """
        loss = 0
        for rewards, log_probs, state in self.buffer:  # iterate over all stored trajectories
            # compute reward for a greedy policy
            greedy_policy_length, _ = greedy_policy(state["nodes"])

            returns = []
            G = 0
            for r in rewards[::-1]:
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)

            log_probs = torch.stack(log_probs)
            score = G + greedy_policy_length
            loss -= torch.sum(log_probs) * score
            self.scores.append(score.item())
        self.optimizer.zero_grad()
        (loss / len(self.buffer)).backward()  # performs backpropagation, i.e. computes the gradients
        self.optimizer.step()
        self.buffer = []  # reset the buffer
        

    def train(self):
        """The train method calls the other methods (data collection and update) and save some statistics
        and logs for monitoring the training and the parameters of the trained agent (checkpoint)."""
        with tqdm(total=self.training_epochs, position=0, desc="Epoch") as pbar:
            for epoch_num in range(self.training_epochs):
                for _ in tqdm(range(self.episodes_per_epoch), desc="Episode", position=1, leave=False):
                    self.collect_one_episode()
                self.update()
                pbar.update(1)
                # save the scores in a file for plotting them
                with open("scores.txt", "w") as file:
                    file.write("\n".join(str(x) for x in self.scores))
                last_score = sum(self.scores[-self.episodes_per_epoch:])
                mean_score = last_score / self.episodes_per_epoch
                tqdm.write(f"Epoch {epoch_num + 1}, score: {mean_score:.2f}")
                if epoch_num % 100 == 0 or epoch_num == self.training_epochs - 1:
                    self.test()
                    pygame.image.save(self.env.screen, f"epoch_{epoch_num}.jpeg")
                    self.env.close()

                    # save a checkpoint for the agent DNN
                    torch.save(self.agent.state_dict(), "policy.pt")

    def test(self):
        state = self.env.reset()
        done = False
        
        while not done:
            action_probs = self.agent(state)
            action = torch.multinomial(action_probs, 1).item()
            state, _, done, _ = self.env.step(action)
            self.env.render()
        


if __name__ == "__main__":
    reinforce = REINFORCE(n_nodes=6)
    reinforce.train()

    from matplotlib import pyplot as plt
    plt.plot(reinforce.scores)
    plt.savefig("reinforce.png")
    plt.show()