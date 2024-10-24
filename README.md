# Neural Combinatorial Optimization: a tutorial
This repo is for the paper Neural Combinatorial Optimization: a tutorial.

In this repo, the first example presented in the paper, that is the one that uses an MLP agent to create a heuristic for
the TSP is developed.

## Set up
This repository uses Python 3.12.7. Follow the official Python documentation to install this version of Python before moving on.

After installing Python, you need to install all the dependencies (better in a virtual environment):

**(Optional) Create the virtual environment and activate it**

```bash
python3.12 -m venv .venv
```

On Linux:
```bash
source .venv/bin/activate
```

On Windows:
```bat
.venv\Scripts\activate.bat
```

**Install the required packages**
```bash
pip install -r requirements.txt
```

## Test the components
To allow for easy understanding, we created a Python file for each component, namely the Environment, the Agent and the
REINFORCE algorithm. We also have a Python file where we defined a greedy and a random policy,
used as baselines in the REINFORCE algorithm.

Each component can be tested by calling
```bash
python <filename>.py
```

### Environment
In the environment, we code the simulator that, at each episode generate a random instance with `n_nodes` in the [0, 1]² square.

The environment is written using the **Gymnasium** package, where the user only needs to define the `reset` and `step` method in order to define its custom environment.

The `reset` method is responsible for generating new instances whenever an episode ends, while the `step` method is responsible for updating the state and computing the reward when the agent sends an action.

Testing the environment, i.e. calling

```bash
python environment.py
```

creates a random instance with 10 nodes, and an agent acting according to a random policy, and show a video of the agent interacting with the environment. Also, the final reward (i.e. the negative of the length of the tour) gets printed.

In the rendered video, the current node is highlighted in red and whenever the agent acts, the edge between the current node and the next node is drawn.

The user might want to change the code below `if __name__ == "__main__":` in order to interact with the example.

### Agent
The agent is composed of the Deep Neural Network definition (that is an MLP) and of the forward method that takes as input the state of the environment and process it according to what is described in the reference paper.

Namely, the nodes features are concatenated (the O matrix is flattened) and the first and current node features are appended.
Moreover, the masking vector is computed and applied before the *softmax* activation function.

The reason why this masking is equivalent to the one in the paper, is explained in the paper appendix.

When calling
```bash
python agent.py
```
a random instance with 6 nodes is spawned and the MLP agent interacts with it.

If no checkpoint model is provided, then the agent will act randomly.
Otherwise, if the user provides a checkpoint model (for example, the `policy.py` file already in the repository), then the agent will act according to the training received.

To check the confidence of the agent before and after training, at each action, the policy probabilities are printed.

## Baselines
The baselines are static policies (i.e. they can not be trained) used to compare the performance of our agent.

We developed two baselines: the random policy and the greedy policy.
The first one select the next node to visit randomly but ensuring the feasibility of the solution; the second policy always select the closest feasible node (at the first step, it always selects the first node in the list).

### REINFORCE
The REINFORCE algorithm is the core of this repository, where the training for the agent happens.

In the `reinforce.py` file, all the pieces (environment, agent, baseline policy) are used to perform both the data collection and the update of the DNN parameters.

## Philosophy
The code is highly commented, and we preferred simplicity and clarity over code performances and best practices.

For this reason, training takes a lot of time and while the agent performance improves, it does not reach satisfactory results in a reasonable amount of time.

### Results
We tried to run the code on an **Intel® Core™ i7-8665U CPU @ 1.90GHz × 8** machine.

You can replicate this run by simply typing
```bash
python reinforce.py
```
in your terminal.

After 2 days of run, we got the version of the DNN parameters that you can find in this repo as `policy.pt`.

We also collected the *scores* of the agent during training, where the score is the difference between the agent reward and the baseline policy reward (a positive score means that the agent performed better than the baseline).

The plot of the scores is the following:

![Plot for the scores of the REINFORCE algorithm](./scores.png)

We can see that the agent was still learning. The baseline policy used is the greedy policy.
Hence, we can see that slowly over time, the agent is learning how to perform better than the greedy policy.

The increase in performance is better visualized using the running average over 1000 episodes:

![Plot of the running average scores](./running_avg_scores.png)

Finally, we can compare some solutions given at the first training iterations with some solutions computed at the end of the training.

**Start of the training**          |  **End of the training**
:-------------------------:|:-------------------------:
Epoch 0 ![](./epoch_0.jpeg)  | Epoch 6900 ![](./epoch_6900.jpeg)
Epoch 700 ![](./epoch_700.jpeg) | Epoch 7000 ![](./epoch_7000.jpeg)

As one can see, there is an improvement in how well the agent performs, while it is still far from reaching optimal solutions for all instances.

#### Disclaimer
We know that these results are not promising for someone approaching NCO for the first time.

A training time of 2 days was not able to produce a heuristic that consistently outperforms the greedy policy.

But there are a few things to consider. First, as outlined at the start of this section, the philosophy guiding this repository was clarity and simplicity, not performances.

Second, the DNN used and the algorithm used are not state of the art, and are quite simple.

Third, the agent samples action according to the policy. Sometimes it might happen that the agent selects a bad action instead of the most promising one.

There are a number of improvements that one can make:

* Parallelize the environment: let the same agent play more instances in parallel to accelerate the data collection phase.
* Use SotA agents and algorithms.
* Train the agent using GPUs.
* Improve the sampling strategy while testing the agent.

When all the above steps are done, there is a huge improvement in the results.

The interested reader might check out the Attention Model developed in the [RL4CO](https://github.com/ai4co/rl4co) repository. There, by running their [Quickstart notebook](https://github.com/ai4co/rl4co/blob/main/examples/1-quickstart.ipynb) (that can run on Colab), one can see that after only 3 epochs of training on a TSP instance with 50 nodes, the Attention Model agent consistently outperforms the greedy policy.