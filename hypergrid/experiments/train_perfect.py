from collections import Counter

import numpy as np
import torch

try:
    import wandb
except ModuleNotFoundError:
    pass
from tqdm import tqdm, trange
from gfn.env import Env

from ml_collections.config_dict import ConfigDict


def validate_perfect(
    env: Env,
    visited_terminating_states: torch.Tensor,
    n_validation_samples: int = 20000,
):
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

    total_states_indices = torch.cat(visited_terminating_states, dim=0)
    states_indices = total_states_indices[-n_validation_samples:].cpu().numpy().tolist()

    counter = Counter(states_indices)
    counter_list = [counter[state_idx] if state_idx in counter else 0 for state_idx in range(env.n_terminating_states)]
    final_states_dist_pmf = torch.tensor(counter_list, dtype=torch.float) / len(states_indices)

    true_dist_pmf = env.true_dist_pmf

    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    l2_dist = ((final_states_dist_pmf - true_dist_pmf) ** 2).sum().sqrt().item()
    kl_dist = (true_dist_pmf * torch.log(true_dist_pmf / (final_states_dist_pmf + 1e-9))).sum().item()
    validation_info = {"l1_dist": l1_dist, "l2_dist": l2_dist, "kl_dist": kl_dist, "final_states_dist_pmf": final_states_dist_pmf}
    return validation_info


def train_perfect(env: Env, experiment_name: str, general_config: ConfigDict, algo_config: ConfigDict):

    use_wandb = len(general_config.wandb_project) > 0
    visited_terminating_states = []

    states_visited = 0
    (
        kl_history,
        l1_history,
        l2_history,
        final_states_dist_pmf_history,
        nstates_history,
        fair_kl_history,
        fair_l1_history,
        fair_l2_history,
        fair_final_states_dist_pmf_history,
    ) = ([], [], [], [], [], [], [], [], [])

    # Train loop
    n_iterations = general_config.n_trajectories // general_config.n_envs
    for iteration in trange(n_iterations):
        true_dist = torch.distributions.Categorical(probs=env.true_dist_pmf)
        visited_states = true_dist.sample((general_config.n_envs,))
        visited_terminating_states.append(visited_states)

        states_visited += general_config.n_envs

        to_log = {"states_visited": states_visited}

        if use_wandb:
            wandb.log(to_log, step=iteration)

        if (iteration + 1) % general_config.validation_interval == 0:
            validation_info = validate_perfect(
                env,
                visited_terminating_states,
                general_config.validation_samples,
            )
            to_log.update(validation_info)

            if use_wandb:
                wandb.log(validation_info, step=iteration)
            tqdm.write(f"{iteration}: {to_log}")

            kl_history.append(to_log["kl_dist"])
            l1_history.append(to_log["l1_dist"])
            l2_history.append(to_log["l2_dist"])
            nstates_history.append(to_log["states_visited"])
            final_states_dist_pmf_history.append(to_log["final_states_dist_pmf"])

            np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
            np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
            np.save(f"{experiment_name}_l2.npy", np.array(l2_history))
            np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))
            np.save(f"{experiment_name}_final_states_dist_pmf.npy", np.array(final_states_dist_pmf_history))

            if "fair_validation_samples" in general_config.keys() and general_config.fair_validation_samples > 0:
                current_visited_terminating_states = [
                    true_dist.sample((general_config.n_envs,)) for _ in range(len(visited_terminating_states))
                ]
                fair_validation_info = validate_perfect(
                    env,
                    current_visited_terminating_states,
                    general_config.fair_validation_samples,
                )
                to_log.update({"fair_" + k: v for k, v in fair_validation_info.items()})

                fair_kl_history.append(to_log["fair_kl_dist"])
                fair_l1_history.append(to_log["fair_l1_dist"])
                fair_l2_history.append(to_log["fair_l2_dist"])
                fair_final_states_dist_pmf_history.append(to_log["fair_final_states_dist_pmf_dist"])

                np.save(f"{experiment_name}_fair_kl.npy", np.array(fair_kl_history))
                np.save(f"{experiment_name}_fair_l1.npy", np.array(fair_l1_history))
                np.save(f"{experiment_name}_fair_l2.npy", np.array(fair_l2_history))
                np.save(f"{experiment_name}_fair_final_states_dist_pmf.npy", np.array(fair_final_states_dist_pmf_history))
