from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torchtyping import TensorType as TT

from collections import Counter
from typing import Dict, Optional

from gfn.env import Env
from gfn.gflownet import GFlowNet, TBGFlowNet
from gfn.modules import GFNModule
from gfn.states import States

from torch.distributions import Categorical, Distribution
from torchtyping import TensorType as TT

from gfn.env import DiscreteEnv, Env
from gfn.states import DiscreteStates, States
from gfn.utils.distributions import UnsqueezedCategorical


def get_terminating_state_dist_pmf(env: Env, states: States) -> TT["n_states", float]:
    states_indices = env.get_terminating_states_indices(states).cpu().numpy().tolist()
    counter = Counter(states_indices)
    counter_list = [counter[state_idx] if state_idx in counter else 0 for state_idx in range(env.n_terminating_states)]

    return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)


def validate(
    env: Env,
    gflownet: GFlowNet,
    n_validation_samples: int = 20000,
    visited_terminating_states: Optional[States] = None,
) -> Dict[str, float]:
    """Evaluates the current gflownet on the given environment.

    This is for environments with known target reward. The validation is done by
    computing the l1 distance between the learned empirical and the target
    distributions.

    Args:
        env: The environment to evaluate the gflownet on.
        gflownet: The gflownet to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training. If given, the pmf is obtained from
            these last n_validation_samples states. Otherwise, n_validation_samples are resampled for evaluation.

    Returns: A dictionary containing the l1 validation metric. If the gflownet
        is a TBGFlowNet, i.e. contains LogZ, then the (absolute) difference
        between the learned and the target LogZ is also returned in the dictionary.
    """

    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
    if isinstance(true_dist_pmf, torch.Tensor):
        true_dist_pmf = true_dist_pmf.cpu()
    else:
        # The environment does not implement a true_dist_pmf property, nor a log_partition property
        # We cannot validate the gflownet
        return {}

    logZ = None
    if isinstance(gflownet, TBGFlowNet):
        logZ = gflownet.logZ.item()
    if visited_terminating_states is None:
        terminating_states = gflownet.sample_terminating_states(n_validation_samples)
    else:
        terminating_states = visited_terminating_states[-n_validation_samples:]

    final_states_dist_pmf = get_terminating_state_dist_pmf(env, terminating_states)

    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    kl_dist = (true_dist_pmf * torch.log(true_dist_pmf / (final_states_dist_pmf + 1e-9))).sum().item()
    validation_info = {"l1_dist": l1_dist, "kl_dist": kl_dist}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)
    return validation_info


def recurcive_logNs_calculation(logNs: torch.tensor, is_calculated: torch.tensor, coord: torch.tensor):
    ancestors_logNs = []
    for i in range(len(logNs.shape)):
        if coord[i] > 0:  # which means that the transition is possible
            ancestor_coord = coord.clone()
            ancestor_coord[i] -= 1
            if not is_calculated[tuple(ancestor_coord)]:
                recurcive_logNs_calculation(logNs, is_calculated, ancestor_coord)
            ancestors_logNs.append(logNs[tuple(ancestor_coord)])
    is_calculated[tuple(coord)] = True
    logNs[tuple(coord)] = torch.logsumexp(torch.tensor(ancestors_logNs), 0)


class DiscreteMaxEntBackwardEstimator(GFNModule):
    r"""Container for forward and backward policy estimators.

    $s \mapsto (P_F(s' \mid s))_{s' \in Children(s)}$.

    or

    $s \mapsto (P_B(s' \mid s))_{s' \in Parents(s)}$.

    Note that while this class resembles LogEdgeFlowProbabilityEstimator, they have
    different semantic meaning. With LogEdgeFlowEstimator, the module output is the log
    of the flow from the parent to the child, while with DiscretePFEstimator, the
    module output is arbitrary.

    Attributes:
        temperature: scalar to divide the logits by before softmax.
        sf_bias: scalar to subtract from the exit action logit before dividing by
            temperature.
        epsilon: with probability epsilon, a random action is chosen.
    """

    def __init__(
        self,
        env: Env,
        module: nn.Module,
    ):
        """Initializes a estimator for P_F for discrete environments.

        Args:
            forward: if True, then this is a forward policy, else backward policy.
            greedy_eps: if > 0 , then we go off policy using greedy epsilon exploration.
            temperature: scalar to divide the logits by before softmax. Does nothing
                if greedy_eps is 0.
            sf_bias: scalar to subtract from the exit action logit before dividing by
                temperature. Does nothing if greedy_eps is 0.
            epsilon: with probability epsilon, a random action is chosen. Does nothing
                if greedy_eps is 0.
        """
        super().__init__(env, module)
        self._forward = False

        shape = [env.height for _ in range(env.ndim)]
        self.logNs, is_calculated = torch.empty(shape), torch.zeros(shape, dtype=torch.bool)
        initial_coord = tuple([0 for _ in range(env.ndim)])
        self.logNs[initial_coord] = 0
        is_calculated[initial_coord] = True
        recurcive_logNs_calculation(self.logNs, is_calculated, torch.tensor(shape) - 1)

    def expected_output_dim(self) -> int:
        if self._forward:
            return self.env.n_actions
        else:
            return self.env.n_actions - 1

    def to_probability_distribution(
        self,
        states: DiscreteStates,
        module_output: TT["batch_shape", "output_dim", float],
    ) -> Categorical:
        """Returns a probability distribution given a batch of states and module output."""
        masks = states.backward_masks
        states.backward_masks.to(torch.int)
        logits = torch.zeros_like(states.tensor, dtype=torch.float)
        for i in range(logits.shape[0]):
            state = states[i].tensor
            for d in range(logits.shape[1]):
                if masks[i, d]:
                    prev_state = state.clone()
                    prev_state[d] -= 1
                    logits[i, d] = self.logNs[tuple(prev_state)] - self.logNs[tuple(state)]
                else:
                    logits[i, d] = -torch.inf
        assert torch.allclose(logits.exp().sum(-1), torch.ones(1))
        return UnsqueezedCategorical(logits=logits)
