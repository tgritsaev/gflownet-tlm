# Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization

Official repository for the paper [Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization](https://arxiv.org/abs/2410.15474).

Timofei Gritsaev, Nikita Morozov, Sergei Samsonov, Daniil Tiapkin.

## Abstract
Generative Flow Networks (GFlowNets) are a family of generative models that learn to sample objects with probabilities proportional to a given reward function. 
The key concept behind GFlowNets is the use of two stochastic policies: a forward policy, which incrementally constructs compositional objects, and a backward policy, which sequentially deconstructs them. 
Recent results show a close relationship between GFlowNet training and entropy-regularized reinforcement learning (RL) problems with a particular reward design. 
However, this connection applies only in the setting of a fixed backward policy, which might be a significant limitation. 
As a remedy to this problem, we introduce a simple backward policy optimization algorithm that involves direct maximization of the value function in an entropy-regularized Markov Decision Process (MDP) over intermediate rewards. 
We provide an extensive experimental evaluation of the proposed approach across various benchmarks in combination with both RL and GFlowNet algorithms and demonstrate its faster convergence and mode discovery in complex environments.

## Installation

- Create conda environment:

```sh
conda create -n gflownet-tlm python=3.10
conda activate gflownet-tlm
```

- Install dependencies for bitseq and hypergrid experiments:

```sh
pip install -r requirements.txt
```

- Install dependencies for molecular experiments:
```sh
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

## Hypergrids

The code for this section is based on the open repository (https://github.com/d-tiapkin/gflownet-rl) and extensively uses the `torchgfn` library (https://github.com/GFNOrg/torchgfn).

Path to configurations (utilizes `ml-collections` library):
- General configuration: `hypergrid/experiments/config/general.py`
- Algorithm: `hypergrid/experiments/config/algo.py`
- Environment: `hypergrid/experiments/config/hypergrid.py`

Available options:
- List of available algorithms: `db`, `tb`, `subtb`, `soft_dqn`, and `munchausen_dqn`;
- List of available backward approaches: `uniform`, `naive`, `maxent`, and `tlm`.

An example how to run the experiment from the `hypergrid/` directory with `standard` rewards, seed `3` on the algorithm `munchausen_dqn`:
```bash
python run_hypergrid_exp.py --general experiments/config/general.py:3 --env experiments/config/hypergrid.py:standard --algo experiments/config/algo.py:munchausen_dqn --algo.backward_approach tlm
```
To train with `uniform` backward policy for this setting:
```bash
python run_hypergrid_exp.py --general experiments/config/general.py:3 --env experiments/config/hypergrid.py:standard --algo experiments/config/algo.py:munchausen_dqn --algo.backward_approach uniform
```

## Bit sequences is COMING SOON

## Molecules

The experiments with molecular environments leverage the existing codebase for molecule generation using GFlowNets (available at https://github.com/recursionpharma/gflownet), which is distributed under the MIT license.

- List of available algorithms: `db`, `tb`, `subtb`, and `dqn`;
- List of available backward approaches: `uniform`, `naive`, `maxent`, and `tlm`.

To reproduce the QM9 experiment with `Munchausen DQN`, learning rate `5e-4`, and backward approach `TLM`:
```bash
python mols.tasks.qm9 --seed 1 --algo dqn --lr 5e-4 --backward_approach tlm
```
To run the sEH experiment with the same hyperparameters:
```bash
python mols.tasks.seh --seed 1 --algo dqn --lr 5e-4 --backward_approach tlm
```