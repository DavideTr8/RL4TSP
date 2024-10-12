import gymnasium as gym
import numpy as np
import pygame

class TSPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, n_nodes, render_mode=None, screen_size=600, reward_mode="dense", seed=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.reward_mode = reward_mode

        if render_mode is None:
            render_mode = "rgb_array"
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {render_mode}")
        self.render_mode = render_mode

        self.screen_size = screen_size
        self.seed(seed)

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(n_nodes)
        self.observation_space = gym.spaces.Dict({
            "nodes": gym.spaces.Box(low=0, high=1, shape=(2, n_nodes), dtype=np.float32),
            "visited": gym.spaces.Sequence(gym.spaces.Discrete(n_nodes))
        })
        
        self.nodes: np.ndarray = np.random.rand(2, n_nodes)
        self.visited: list[int]


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # self.nodes = np.random.rand(2, self.n_nodes)
        self.visited = []
        return {"nodes": self.nodes, "visited": self.visited}

    def step(self, action: int):
        # Placeholder for step logic
        self.visited.append(action)
        if len(self.visited) == self.n_nodes:
            done = True
        else:
            done = False
        
        if done:
            reward = 0
            for i in range(self.n_nodes - 1):
                reward -= np.linalg.norm(self.nodes[:, self.visited[i]] - self.nodes[:, self.visited[i + 1]])
            reward -= np.linalg.norm(self.nodes[:, self.visited[-1]] - self.nodes[:, self.visited[0]])
        else:
            reward = 0
        info = {}
        next_state = {"nodes": self.nodes, "visited": self.visited}
        return next_state, reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            if not hasattr(self, 'screen'):
                pygame.init()
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.screen.fill((255, 255, 255))
            for node_idx in range(self.n_nodes):
                node = self.nodes[:, node_idx]
                pygame.draw.circle(self.screen, (0, 0, 255), (int(node[0] * self.screen_size), int(node[1] * self.screen_size)), 5)
            for i in range(len(self.visited) - 1):
                pygame.draw.line(self.screen, (0, 0, 0), (int(self.nodes[0, self.visited[i]] * self.screen_size), int(self.nodes[1, self.visited[i]] * self.screen_size)), (int(self.nodes[0, self.visited[i + 1]] * self.screen_size), int(self.nodes[1, self.visited[i + 1]] * self.screen_size)), width=2)
            if len(self.visited) == self.n_nodes:
                pygame.draw.line(self.screen, (0,0,0), (int(self.nodes[0, self.visited[-1]] * self.screen_size), int(self.nodes[1, self.visited[-1]] * self.screen_size)), (int(self.nodes[0, self.visited[0]] * self.screen_size), int(self.nodes[1, self.visited[0]] * self.screen_size)), width=2)
            # change color of the last node in visited to red
            if len(self.visited) > 0:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(self.nodes[0, self.visited[-1]] * self.screen_size), int(self.nodes[1, self.visited[-1]] * self.screen_size)), 5)
            pygame.display.flip()
        elif mode == "rgb_array":
            # Return an RGB array representation of the state
            pass

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()


if __name__ == "__main__":
    n_nodes = 3
    env = TSPEnv(n_nodes)
    env.reset()
    done = False
    import time
    for i in range(n_nodes):
        action = i
        next_state, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5)
    print(reward)
    env.close()