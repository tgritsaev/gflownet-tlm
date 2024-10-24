import copy
import os
import pathlib

import git
import torch
from omegaconf import OmegaConf
from torch import Tensor
import torch_geometric.data as gd

from mols.algo.advantage_actor_critic import A2C
from mols.algo.flow_matching import FlowMatching
from mols.algo.soft_dqn import SoftDQN
from mols.algo.soft_q_learning import SoftQLearning
from mols.algo.trajectory_balance import TrajectoryBalance
from mols.data.data_source import DataSource
from mols.data.replay_buffer import ReplayBuffer
from mols.models.graph_transformer import GraphTransformerGFN

from .trainer import GFNTrainer


def model_grad_norm(model):
    x = 0
    for i in model.parameters():
        if i.grad is not None:
            x += (i.grad * i.grad).sum()
    return torch.sqrt(x)


class StandardOnlineTrainer(GFNTrainer):
    def setup_model(self):
        self.model = GraphTransformerGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.do_predict_n + 1,
            unif_init=self.cfg.model.unif_init,
        )

    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "TB":
            algo = TrajectoryBalance
        elif algo == "FM":
            algo = FlowMatching
        elif algo == "A2C":
            algo = A2C
        elif algo == "SQL":
            algo = SoftQLearning
        elif algo == "DQN":
            algo = SoftDQN
        else:
            raise ValueError(algo)
        self.algo = algo(self.env, self.ctx, self.cfg)

    def setup_data(self):
        self.training_data = []
        self.test_data = []

    def setup_optimizer(self, params, lr=None, momentum=None):
        if lr is None:
            lr = self.cfg.opt.learning_rate
        if momentum is None:
            momentum = self.cfg.opt.momentum
        if self.cfg.opt.opt == "adam":
            return torch.optim.Adam(
                params,
                lr,
                (momentum, 0.999),
                weight_decay=self.cfg.opt.weight_decay,
                eps=self.cfg.opt.adam_eps,
            )

        raise NotImplementedError(f"{self.cfg.opt.opt} is not implemented")

    def setup(self):
        super().setup()
        self.offline_ratio = 0
        self.replay_buffer = None
        self.sampling_hooks.append(AvgRewardHook())
        self.valid_sampling_hooks.append(AvgRewardHook())

        def is_any_ban_word_in_parameter_name(ban_words_list, parameter_name):
            for ban_word in ban_words_list:
                if ban_word in parameter_name:
                    return True
            return False

        # create optimizers and lr_schedulers
        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model._logZ.parameters()) if hasattr(self.model, "_logZ") else []
        if self.backward_approach == "tlm":
            not_forward_param_names = ["logZ", "remove_node", "remove_edge_attr"]
            params = [
                p
                for n, p in self.model.named_parameters()
                if not is_any_ban_word_in_parameter_name(not_forward_param_names, n)
            ]
            not_backward_param_names = ["logZ", "stop", "add_node", "set_edge_attr"]
            backward_params = [
                p
                for n, p in self.model.named_parameters()
                if not is_any_ban_word_in_parameter_name(not_backward_param_names, n)
            ]
        else:
            params = [p for n, p in self.model.named_parameters() if (not "logZ" in n)]

        self.optimizer = self.setup_optimizer(params)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 2 ** (-step / self.cfg.opt.lr_decay)
        )
        self.optimizer_Z = self.setup_optimizer(Z_params, self.cfg.algo.tb.Z_learning_rate, 0.9)
        self.lr_scheduler_Z = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_Z, lambda step: 2 ** (-step / self.cfg.algo.tb.Z_lr_decay)
        )
        if self.backward_approach == "tlm":
            self.b_optimizer = self.setup_optimizer(backward_params, self.cfg.opt.backward_learning_rate)
            self.blr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.b_optimizer, lambda step: 2 ** (-step / self.cfg.opt.backward_lr_decay)
            )

        self.sampling_tau = self.cfg.algo.sampling_tau
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

        clip_value = self.cfg.opt.clip_grad_param
        self.clip_grad_callback = {
            "value": lambda params: torch.nn.utils.clip_grad_value_(params, clip_value),
            "norm": lambda params: [torch.nn.utils.clip_grad_norm_(p, clip_value) for p in params],
            "total_norm": lambda params: torch.nn.utils.clip_grad_norm_(params, clip_value),
            "none": lambda x: None,
        }[self.cfg.opt.clip_grad_type]

        # saving hyperparameters
        # try:
        #     self.cfg.git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        # except git.InvalidGitRepositoryError:
        #     self.cfg.git_hash = "unknown"  # May not have been installed through git

        yaml_cfg = OmegaConf.to_yaml(self.cfg)
        if self.print_config:
            print("\n\nHyperparameters:\n")
            print(yaml_cfg)
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        with open(pathlib.Path(self.cfg.log_dir) / "config.yaml", "w", encoding="utf8") as f:
            f.write(yaml_cfg)

    def step(self, batch: gd.Batch, train_it: int):
        
        def update_forward_policy(batch):
            loss, info = self.algo.compute_batch_losses(
                self.model,
                batch,
                self.sampling_model if self.use_sampling_model else None,
            )
            tlm = info["tlm"]
            if not torch.isfinite(loss):
                self.inf_loss_cnt += 1
                if self.inf_loss_cnt > 200:
                    raise ValueError("loss is not finite")
                return {"grad_norm": 0, "grad_norm_clip": 0}
            # print("1. loss, tlm are computed")
            loss.backward()

            self.optimizer.step()
            self.optimizer_Z.step()
            # print("1.2. step()")
            with torch.no_grad():
                g0 = model_grad_norm(self.model)
                self.clip_grad_callback(self.model.parameters())
                g1 = model_grad_norm(self.model)
                info.update({"grad_norm": g0, "grad_norm_clip": g1})
            self.lr_scheduler.step()
            self.lr_scheduler_Z.step()
            self.optimizer.zero_grad()
            self.optimizer_Z.zero_grad()
            
            return info
            
        def update_backward_policy_if_required(batch):
            if self.backward_approach == "tlm":
                # print("2. backward learning")
                loss, info2 = self.algo.compute_batch_losses(
                    self.model,
                    batch,
                )
                tlm = info2["tlm"]
                if not torch.isfinite(tlm):
                    self.inf_loss_cnt += 1
                    if self.inf_loss_cnt > 200:
                        raise ValueError("tlm is not finite")
                    return {"grad_norm": 0, "grad_norm_clip": 0, "bgrad_norm": 0, "bgrad_norm_clip": 0}
                if self.backward_approach == "naive":
                    # print("2.1.a loss.backward()")
                    loss.backward()
                elif self.backward_approach == "tlm":
                    # print("2.1.b tlm.backward()")
                    tlm.backward()
                else:
                    raise KeyError
                # with torch.no_grad():
                #     g0 = model_grad_norm(self.model)
                #     self.clip_grad_callback(self.model.parameters())
                #     g1 = model_grad_norm(self.model)
                #     info.update({"bgrad_norm": g0, "bgrad_norm_clip": g1})
                self.b_optimizer.step()
                self.blr_scheduler.step()
                self.b_optimizer.zero_grad()
                
            if self.sampling_tau > 0:
                for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                    b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))
                
        if not self.reverse_updates_order:
            info = update_forward_policy(batch)
            update_backward_policy_if_required(batch)
        else:
            update_backward_policy_if_required(batch)
            info = update_forward_policy(batch)
            

        info["lr"] = self.lr_scheduler.get_last_lr()[0]
        if self.backward_approach == "tlm":
            info["blr"] = self.blr_scheduler.get_last_lr()[0]

        return info


class AvgRewardHook:
    def __call__(self, trajs, rewards, obj_props, extra_info):
        return {"sampled_reward_avg": rewards.mean().item()}
