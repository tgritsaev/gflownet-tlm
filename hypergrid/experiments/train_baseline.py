r"""
Adapted from torchgfn library
https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid.py
"""

import os
import torch
from copy import deepcopy

# import wandb
from tqdm import tqdm, trange
import numpy as np

from gfn.gflownet import (
    DBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from experiments.utils import DiscreteMaxEntBackwardEstimator, validate
from gfn.utils.modules import DiscreteUniform, NeuralNet
from ml_collections.config_dict import ConfigDict


def train_baseline(
    env: Env,
    experiment_name: str,
    general_config: ConfigDict,
    algo_config: ConfigDict,
):
    use_wandb = len(general_config.wandb_project) > 0
    experiment_name += f"_lr={algo_config.learning_rate}_lrg={algo_config.gamma}"
    experiment_name += f"_{algo_config.backward_approach}"
    if algo_config.backward_approach == "tlm":
        experiment_name += f"_pb_tau={algo_config.pb_tau}"
    if algo_config.first_pf_update:
        experiment_name += f"_first_pf_update"
    print(f"{experiment_name=}", flush=True)

    # Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet = None
    pb_module = None
    # We need a DiscretePFEstimator and a DiscretePBEstimator

    pf_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )
    if algo_config.backward_approach in ["uniform", "maxent"]:
        pb_module = DiscreteUniform(env.n_actions - 1)
    else:
        pb_module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=algo_config.net.hidden_dim,
            n_hidden_layers=algo_config.net.n_hidden,
            torso=pf_module.torso if algo_config.tied else None,
        )
        pb_module.last_layer.weight.data.fill_(0.0)
        pb_module.last_layer.bias.data.fill_(0.0)

    assert pf_module is not None, f"pf_module is None. Command-line arguments: {algo_config}"
    assert pb_module is not None, f"pb_module is None. Command-line arguments: {algo_config}"
    pf_estimator = DiscretePolicyEstimator(env=env, module=pf_module, forward=True)
    if algo_config.backward_approach == "tlm":
        target_pb_module = deepcopy(pb_module)
        target_pb_module.load_state_dict(pb_module.state_dict())
        target_pb_estimator = DiscretePolicyEstimator(env=env, module=target_pb_module, forward=False)
        pb_estimator = DiscretePolicyEstimator(env=env, module=pb_module, forward=False)
    elif algo_config.backward_approach == "maxent":
        target_pb_estimator = DiscreteMaxEntBackwardEstimator(env=env, module=pb_module)
    else:
        target_pb_estimator = DiscretePolicyEstimator(env=env, module=pb_module, forward=False)

    if algo_config.name in ("DetailedBalance", "SubTrajectoryBalance"):
        # We need a LogStateFlowEstimator
        assert pf_estimator is not None, f"pf_estimator is None. Arguments: {algo_config}"
        assert target_pb_estimator is not None, f"target_pb_estimator is None. Arguments: {algo_config}"

        module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=1,
            hidden_dim=algo_config.net.hidden_dim,
            n_hidden_layers=algo_config.net.n_hidden,
            torso=pf_module.torso if algo_config.tied else None,
        )

        logF_estimator = ScalarEstimator(env=env, module=module)
        if algo_config.name == "DetailedBalance":
            gflownet = DBGFlowNet(
                pf=pf_estimator,
                pb=target_pb_estimator,
                logF=logF_estimator,
                on_policy=False,
            )
        else:
            gflownet = SubTBGFlowNet(
                pf=pf_estimator,
                pb=target_pb_estimator,
                logF=logF_estimator,
                on_policy=False,
                weighting=algo_config.subTB_weighting,
                lamda=algo_config.subTB_lambda,
            )
    elif algo_config.name == "TrajectoryBalance":
        gflownet = TBGFlowNet(
            pf=pf_estimator,
            pb=target_pb_estimator,
            on_policy=False,
        )

    # 3. Create the optimizer
    if algo_config.backward_approach == "tlm":
        params_list = [v for k, v in dict(gflownet.named_parameters()).items() if (not "pb" in k and k != "logZ")]
    else:
        params_list = [v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"]
    params = [
        {
            "params": params_list,
            "lr": algo_config.learning_rate,
        }
    ]
    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": algo_config.learning_rate_Z,
            }
        )
    optimizer = torch.optim.Adam(params)

    if algo_config.backward_approach == "tlm":
        pb_params = [
            {
                "params": pb_module.parameters(),
                "lr": algo_config.learning_rate,
            }
        ]
        pb_optimizer = torch.optim.Adam(pb_params)
        pb_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(pb_optimizer, algo_config.gamma)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    kl_history, l1_history, nstates_history, deviation_history = [], [], [], []

    # Train loop
    n_iterations = general_config.n_trajectories // general_config.n_envs

    def update_forward_policy(training_samples):
        optimizer.zero_grad()
        loss = gflownet.loss(training_samples)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def update_backward_policy_if_required(trajectories):
        # with torch.autograd.set_detect_anomaly(True):
        if algo_config.backward_approach == "tlm":
            pb_optimizer.zero_grad()

            # this code is taken from the class TrajectoryBasedGFlowNet(PFBasedGFlowNet)
            # function `get_pfs_and_pbs(...)`
            valid_states = trajectories.states[~trajectories.states.is_sink_state]
            valid_actions = trajectories.actions[~trajectories.actions.is_dummy]

            non_initial_valid_states = valid_states[~valid_states.is_initial_state]
            non_exit_valid_actions = valid_actions[~valid_actions.is_exit]

            module_output = pb_estimator(non_initial_valid_states)
            valid_log_pb_actions = pb_estimator.to_probability_distribution(
                non_initial_valid_states, module_output
            ).log_prob(non_exit_valid_actions.tensor)

            pb_loss = -torch.sum(valid_log_pb_actions)

            pb_loss.backward(retain_graph=True)
            pb_optimizer.step()
            pb_lr_scheduler.step()

            with torch.no_grad():
                for param, target_param in zip(pb_module.parameters(), target_pb_module.parameters()):
                    target_param.data.mul_(1 - algo_config.pb_tau)
                    torch.add(target_param.data, param.data, alpha=algo_config.pb_tau, out=target_param.data)

            pb_optimizer.zero_grad()

    for iteration in range(n_iterations):
        trajectories = gflownet.sample_trajectories(n_samples=general_config.n_envs)
        training_samples = gflownet.to_training_samples(trajectories)

        if algo_config.first_pf_update:
            loss = update_forward_policy(training_samples)
            update_backward_policy_if_required(trajectories)
        else:
            update_backward_policy_if_required(trajectories)
            loss = update_forward_policy(training_samples)

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"loss": loss.item(), "states_visited": states_visited}
        if use_wandb:
            wandb.log(to_log, step=iteration)
        if (iteration + 1) % general_config.validation_interval == 0:
            validation_info = validate(
                env,
                gflownet,
                general_config.validation_samples,
                visited_terminating_states,
            )
            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            # tqdm.write(f"{iteration}: {to_log}")
            print(f"{iteration}: {to_log}")

            kl_history.append(to_log["kl_dist"])
            l1_history.append(to_log["l1_dist"])
            nstates_history.append(to_log["states_visited"])

            if algo_config.backward_approach in ["tlm", "naive"]:
                valid_states = trajectories.states[~trajectories.states.is_sink_state]
                valid_actions = trajectories.actions[~trajectories.actions.is_dummy]

                non_initial_valid_states = valid_states[~valid_states.is_initial_state]
                non_exit_valid_actions = valid_actions[~valid_actions.is_exit]

                module_output = target_pb_estimator(non_initial_valid_states)
                valid_log_pb_actions = target_pb_estimator.to_probability_distribution(
                    non_initial_valid_states, module_output
                ).log_prob(non_exit_valid_actions.tensor)

                deviation_history.append(
                    torch.mean(
                        valid_log_pb_actions - torch.log(1 / torch.count_nonzero(non_initial_valid_states.tensor, -1))
                    ).item()
                )

        if (iteration + 1) % 1000 == 0:
            np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
            np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
            np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))
            np.save(f"{experiment_name}_deviation.npy", np.array(deviation_history))

    np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
    np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
    np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))
    np.save(f"{experiment_name}_deviation.npy", np.array(deviation_history))

    torch.save(gflownet.state_dict(), f"{experiment_name}_gfn.pt")
