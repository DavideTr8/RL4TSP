import torch
import torch.nn as nn

class MLPAgent(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_nodes * 2 + 4, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, n_nodes),
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, state):
        nodes = torch.from_numpy(state["nodes"]).float()
        if len(state["visited"]) > 0:
            first_node = nodes[:, state["visited"][0]]
            current_node = nodes[:, state["visited"][-1]]
        else:
            first_node = -torch.ones(2)
            current_node = -torch.ones(2)

        mlp_input = torch.cat([nodes.T.flatten(), first_node, current_node]).unsqueeze(0)            
        probabilities = self.model(mlp_input)

        mask = torch.zeros_like(probabilities)
        mask[:, state["visited"]] = -1e6
        probabilities = self.softmax(probabilities + mask)
        return probabilities
    


if __name__ == "__main__":
    from gym_env import TSPEnv
    import time
    n_nodes = 4
    agent = MLPAgent(n_nodes)
    env = TSPEnv(n_nodes)

    done = False
    state = env.reset()
    while not done:
        probs = agent(state)
        action = torch.multinomial(probs, 1).item()
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.3)
    print(env.visited)
    print(reward)