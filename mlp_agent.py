import torch
import torch.nn as nn
import numpy as np

class MLPAgent(nn.Module):
    """RL agent that uses an MLP network to decide the probabilities to assign to each action.
    
    It is composed by an input Linear layer that takes as input the xy-coordinates of the nodes of the problem,
    and the xy-coordinates of the first visited node and current node.
    Hence, the length of the input vector is n_nodes * 2 + 2*2. The output dimension is hidden_dim.
    After the input layer, is an activation layer, where the activation function is the Tanh function.
    Then there is another Linear layer. This is a hidden layer with input hidden_dim and output hidden_dim.
    This is followed by another activation layer with Tanh function.
    Finally, the there is a Linear layer that goes from dimension 64 to dimension n_nodes (number of actions).
    As last is the softmax layer, that is applied after masking (instead of before as explained in the paper) 
    for numerical reasons.
    """
    def __init__(self, n_nodes: int, hidden_dim: int = 64):
        """Method that initializes the DNN.
        
        `n_nodes` is the number of nodes in each instance. This value is used to know the dimension of the 
        input vector and the dimension of the output vector (that must match the number of actions).
        `hidden_dim` is the dimension of the hidden layers. All the hidden layers have the same dimension.
        
        torch.nn.Sequential creates a sequence of layers, similarly as a composition of functions.
        torch.nn.Linear is a Linear layer, that is equivalent to a matrix multiplication.
        torch.nn.Tanh() is the hyperbolic tangent function (https://en.wikipedia.org/wiki/Hyperbolic_functions).
            Such function is applied pointwise to each entry of the hidden vector.
        torch.nn.Softmax is the softmax function, as defined in the paper. For those unfamiliar with Pytorch, 
            the parameter `dim=-1` for the softmax can be ignored.
        """
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_nodes * 2 + 4, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, n_nodes),
        )
        self.softmax = torch.nn.Softmax(dim=-1) 

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """The forward function specifies how data are fed to the DNN.
        
        Numpy and Pytorch are two Python packages that handle arrays. Numpy is used for scientific programming, while
        Pytorch for Deep Learning.
        Numpy arrays can be converted to Pytorch arrays (a.k.a. tensors), and this is needed
        since Pytorch DNNs only accept tensors as input but the environment uses Numpy.

        This method creates the input vector starting from the environment state. The input is obtained by concatenating
        all the nodes features in a single vector (i.e. flattening the O_matrix).
        Then, we also concatenate the first and current node features to obtain o_t.
        Finally we create the mask mu_t using l_t and apply it before computing the probabilities.
        Why this is equal to the process described in the paper, check the paper's Appendix.
        """
        O_matrix = torch.from_numpy(state["nodes"]).float()
        l_t = state["visited"]
        if len(l_t) > 0:
            first_node = O_matrix[:, l_t[0]]
            current_node = O_matrix[:, l_t[-1]]
        else:
            # dummy values for the first action. since nodes are in the [0, 1]x[0, 1] square,
            # we know that no node can be (-1, -1)
            first_node = -torch.ones(2)
            current_node = -torch.ones(2)

        o_t = torch.cat([O_matrix.T.flatten(), first_node, current_node]).unsqueeze(0)            
        y_t = self.model(o_t)

        mu_t = torch.zeros_like(y_t)  # creates a vector of 0s with the same length of y_t
        mu_t[:, l_t] = -1e6
        pi_t = self.softmax(y_t + mu_t)
        return pi_t
    


if __name__ == "__main__":
    from gym_env import TSPEnv
    import time
    n_nodes = 10
    agent = MLPAgent(n_nodes)
    env = TSPEnv(n_nodes)

    done = False
    state = env.reset()
    while not done:
        probs: torch.Tensor = agent(state)
        probs_str = ", ".join([f"{x:.2f}" for x in probs.view(-1).tolist()]) # just for printing purposes
        print(f"The probabilities for this action are: [{probs_str}]")
        action = torch.multinomial(probs, 1).item()
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.3)
    print(f"The order of visit for nodes is {env.visited}")
    print(f"The final reward is {reward}, meaning that the total length of the tour is {-reward}")