import os
import json
from copy import deepcopy

import numpy as np
import torch

from tqdm import tqdm, trange


from algorithms import SoftDQNGFlowNet, TorchRLReplayBuffer

from gfn.modules import DiscretePolicyEstimator
from gfn.containers import Trajectories
from experiments.utils import DiscreteMaxEntBackwardEstimator, validate
from gfn.utils.modules import NeuralNet, DiscreteUniform
from gfn.env import Env

from ml_collections.config_dict import ConfigDict


def train_softdqn(
    env: Env,
    experiment_name: str,
    general_config: ConfigDict,
    algo_config: ConfigDict,
):

    if algo_config.replay_buffer.replay_buffer_size == 0:
        experiment_name += f"_nobf"
    elif algo_config.update_frequency > 1:
        experiment_name += f"_freq={algo_config.update_frequency}"

    if algo_config.is_double:
        experiment_name += "_Double"
    if algo_config.munchausen.alpha > 0:
        experiment_name += f"_M_alpha={algo_config.munchausen.alpha}"

    experiment_name += f"_lr={algo_config.learning_rate}_lrg={algo_config.gamma}"
    experiment_name += f"_{algo_config.backward_approach}"
    if algo_config.backward_approach == "tlm" and not algo_config.smooth_pb:
        experiment_name += "_not_smooth"
    if algo_config.first_pf_update:
        experiment_name += f"_first_pf_update"
    print(f"{experiment_name=}", flush=True)

    use_wandb = len(general_config.wandb_project) > 0
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
        # uniform initialization
        pb_module.last_layer.weight.data.fill_(0.0)
        pb_module.last_layer.bias.data.fill_(0.0)

    pf_estimator = DiscretePolicyEstimator(env=env, module=pf_module, forward=True)
    if algo_config.backward_approach == "maxent":
        pb_estimator = DiscreteMaxEntBackwardEstimator(env=env, module=pb_module)
    else:
        pb_estimator = DiscretePolicyEstimator(env=env, module=pb_module, forward=False)

    pf_target = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )
    pf_target_estimator = DiscretePolicyEstimator(env=env, module=pf_target, forward=True)

    replay_buffer_size = algo_config.replay_buffer.replay_buffer_size

    entropy_coeff = 1 / (1 - algo_config.munchausen.alpha)  # to make (1-alpha)*tau=1
    gflownet = SoftDQNGFlowNet(
        q=pf_estimator,
        q_target=pf_target_estimator,
        pb=pb_estimator,
        smooth_pb=(algo_config.backward_approach == "tlm" and algo_config.smooth_pb),
        on_policy=True if replay_buffer_size == 0 else False,
        is_double=algo_config.is_double,
        entropy_coeff=entropy_coeff,
        munchausen_alpha=algo_config.munchausen.alpha,
        munchausen_l0=algo_config.munchausen.l0,
    )

    replay_buffer = None
    if replay_buffer_size > 0:
        replay_buffer = TorchRLReplayBuffer(
            env,
            replay_buffer_size=replay_buffer_size,
            prioritized=algo_config.replay_buffer.prioritized,
            alpha=algo_config.replay_buffer.alpha,
            beta=algo_config.replay_buffer.beta,
            batch_size=algo_config.replay_buffer.batch_size,
        )

    if algo_config.loss_type == "MSE":
        loss_fn = torch.nn.MSELoss(reduction="none")
    elif algo_config.loss_type == "Huber":  # Used for gradient clipping
        loss_fn = torch.nn.HuberLoss(reduction="none", delta=1.0)
    else:
        raise NotImplementedError(f"{algo_config.loss_type} loss is not supported")

    if algo_config.backward_approach == "tlm":
        pf_params = [
            {
                "params": [
                    v for k, v in dict(gflownet.named_parameters()).items() if (not "pb" in k and k != "q_target")
                ],
                "lr": algo_config.learning_rate,
            }
        ]

        pb_params = [
            {
                "params": gflownet.pb.parameters(),
                "lr": algo_config.learning_rate,
            }
        ]
        pb_optimizer = torch.optim.Adam(pb_params)
        pb_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(pb_optimizer, algo_config.gamma)
    else:
        pf_params = [
            {
                "params": [v for k, v in dict(gflownet.named_parameters()).items() if (k != "q_target")],
                "lr": algo_config.learning_rate,
            }
        ]
        pb_lr_scheduler = None
    optimizer = torch.optim.Adam(pf_params)

    n_iterations = general_config.n_trajectories // general_config.n_envs
    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    kl_history, l1_history, nstates_history = [], [], []

    def update_forward_policy(scores, rb_batch):
        optimizer.zero_grad()
        td_error = loss_fn(scores, torch.zeros_like(scores))
        if replay_buffer is not None and replay_buffer.prioritized:
            replay_buffer.update_priority(rb_batch, td_error.detach())
        loss = td_error.mean()
        loss.backward()
        optimizer.step()
        return loss

    def update_backward_policy_if_required(training_samples):
        if algo_config.backward_approach == "tlm":
            pb_optimizer.zero_grad()
            pb_loss = gflownet.get_pb_loss(training_samples)
            pb_loss.backward()
            pb_optimizer.step()
            pb_lr_scheduler.step()

    # Train loop
    # for iteration in trange(n_iterations):
    for iteration in range(n_iterations):
        progress = float(iteration) / n_iterations
        trajectories = gflownet.sample_trajectories(n_samples=general_config.n_envs)
        training_samples = gflownet.to_training_samples(trajectories)

        update_model = iteration > algo_config.learning_starts and iteration % algo_config.update_frequency == 0
        if update_model and not algo_config.first_pf_update:
            update_backward_policy_if_required(training_samples)

        if replay_buffer is not None:
            with torch.no_grad():
                # For prioritized RB
                if replay_buffer.prioritized:
                    scores = gflownet.get_scores(training_samples)
                    td_error = loss_fn(scores, torch.zeros_like(scores))
                    replay_buffer.add(training_samples, td_error)
                    # Annealing of beta
                    replay_buffer.update_beta(progress)
                else:
                    replay_buffer.add(training_samples)

            if iteration > algo_config.learning_starts:
                training_objects, rb_batch = replay_buffer.sample() 
                scores = gflownet.get_scores(training_objects)
        else:
            training_objects = training_samples
            scores = gflownet.get_scores(training_objects)

        if update_model:
            if algo_config.first_pf_update:
                loss = update_forward_policy(scores, rb_batch)
                update_backward_policy_if_required(training_samples)
            else:
                loss = update_forward_policy(scores, rb_batch)

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {
            "states_visited": states_visited,
            "blr": pb_lr_scheduler.get_last_lr()[0] if pb_lr_scheduler is not None else 0,
        }
        if iteration > algo_config.learning_starts and iteration % algo_config.update_frequency == 0:
            if iteration % algo_config.target_network_frequency == 0:
                gflownet.update_target_nets(algo_config.tau)
            to_log.update({"loss": loss.item()})

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

            kl_history.append(to_log["kl_dist"])
            l1_history.append(to_log["l1_dist"])
            nstates_history.append(to_log["states_visited"])

            # tqdm.write(f"{iteration}: {to_log}")
            print(f"{iteration}: {to_log}")

        if (iteration + 1) % 1000 == 0:
            np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
            np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
            np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))

    np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
    np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
    np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))
