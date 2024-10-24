from agent import MLPAgent
import torch

import random

@torch.no_grad()
def greedy_policy(nodes):
    """Return the length of the tour for the greedy policy."""
    nodes = torch.from_numpy(nodes)
    length = 0
    current_node = nodes[:, 0]
    visited = [0]
    
    # for each node, select the closest node
    for i in range(1, nodes.shape[1]):
        # compute the distance to all the nodes from the current node
        distances = torch.norm(nodes - current_node.unsqueeze(-1), dim=0)
        distances[visited] = 1e6
        # pick the closest node
        closest_node = torch.argmin(distances).item()
        # update the current node
        current_node = nodes[:, closest_node]
        # add the distance to the length
        length += distances[closest_node]
        visited.append(closest_node)
    # add the distance from the last node to the first node
    length += torch.norm(current_node - nodes[:, 0])
    return length, visited

@torch.no_grad()
def random_policy(nodes):
    nodes = torch.from_numpy(nodes)
    # pick nodes at random
    nodes_order = list(range(nodes.shape[1]))
    random.shuffle(nodes_order)

    first_node = nodes[:, nodes_order[0]]
    last_node = nodes[:, nodes_order[-1]]
    length = torch.norm(first_node - last_node, dim=0)
    for idx in range(len(nodes_order) - 1):
        current_idx = nodes_order[idx]
        next_idx = nodes_order[idx + 1]
        current_node = nodes[:, current_idx]
        next_node = nodes[:, next_idx]
        length += torch.norm(next_node - current_node, dim=0)

    return length, nodes_order
    

if __name__ == "__main__":
    from environment import TSPEnv
    import time

    n_nodes = 4
    env = TSPEnv(n_nodes)
    agent = MLPAgent(n_nodes)
    done = False
    state = env.reset()
    while not done:
        probs = agent(state)
        action = torch.multinomial(probs, 1).item()
        state, reward, done, _ = env.step(action)
    
    greedy_policy_length, greedy_visited = greedy_policy(env.nodes)
    random_policy_length, random_visited = random_policy(env.nodes)
    print("Greedy policy length:", greedy_policy_length)
    print("Random policy length:", random_policy_length)
    print("Agent policy length:", -reward)

    # add the greedy policy tour to the render in red
    # import pygame
    # screen = env.screen
    # lines_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    # lines_surface = lines_surface.convert_alpha()
    # lines_surface.fill((0, 0, 0, 0))

    time.sleep(1.0)