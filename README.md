# Neural Combinatorial Optimization: a tutorial
This repo is for the paper Neural Combinatorial Optimization: a tutorial.

In this repo the first example that uses an MLP agent to create an heuristic for the TSP is developed.

## Set up
This repository uses Python 3.12.7. Follow the official Python documentation to install this verion of Python before moving on.

To use the repo, you now need to install all the dependecies (better in in a virtual environment):

**(Optional) Create the virtual environment and activate it**
```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

**Install the required packages**
```bash
pip install -r requirements.txt
```

## Test the components
To allow for easy understanding, we created a Python file for each component, namely the Environment, the Agent and the REINFORCE algorithm. We also have a Python file where we defined a greedy and a random policy,
used as baselines in the REINFORCE algorithm.

Each component can be tested by calling
```bash
python <filename.py>
```

### Philosophy
The code is highly commented and we preferred simplicity and clarity over code performances and best practices.
