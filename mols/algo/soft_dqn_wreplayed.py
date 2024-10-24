import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from itertools import chain

import numpy as np
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter


from mols import GFNAlgorithm
from mols.algo.config import Backward, LossFN, NLoss
from mols.algo.graph_sampling import GraphSampler
from mols.config import Config
from mols.envs.graph_building_env import (
    ActionIndex,
    GraphAction,
    GraphActionCategorical,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    generate_forward_trajectory,
)
from mols.models.graph_transformer import GraphTransformerGFN
from mols.utils.misc import get_worker_device

# it must be done
import multiprocessing as mp

mp.set_start_method("fork")
# otherwise multiprocessing in the data_source.py fails.

from torchrl.data import PrioritizedReplayBuffer, ListStorage


@dataclass
class Transition:
    mol: gd.Data
    next_mol: gd.Data
    is_final_graph: bool
    log_reward: Optional[float]
    cond_info: torch.Tensor
    next_cond_info: torch.Tensor
    actions: ActionIndex
    bck_actions: ActionIndex
    next_bck_actions: ActionIndex
    is_sink: bool
    log_p_B: Optional[torch.Tensor]
    next_log_p_B: Optional[torch.Tensor]
    idx: int
    used: bool = False


def detach_and_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    elif isinstance(x, dict):
        x = {k: detach_and_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        x = [detach_and_cpu(v) for v in x]
    elif isinstance(x, tuple):
        x = tuple(detach_and_cpu(v) for v in x)
    return x


def shift_right(x: torch.Tensor, z=0):
    "Shift x right by 1, and put z in the first position"
    x = torch.roll(x, 1, dims=0)
    x[0] = z
    return x


def cross(x: torch.Tensor):
    """
    Calculate $y_{ij} = \sum_{t=i}^j x_t$.
    The lower triangular portion is the inverse of the upper triangular one.
    """
    assert x.ndim == 1
    y = torch.cumsum(x, 0)
    return y[None] - shift_right(y)[:, None]


def subTB(v: torch.Tensor, x: torch.Tensor):
    r"""
    Compute the SubTB(1):
    $\forall i \leq j: D[i,j] =
        \log \frac{F(s_i) \prod_{k=i}^{j} P_F(s_{k+1}|s_k)}
        {F(s_{j + 1}) \prod_{k=i}^{j} P_B(s_k|s_{k+1})}$
      for a single trajectory.
    Note that x_k should be P_F(s_{k+1}|s_k) - P_B(s_k|s_{k+1}).
    """
    assert v.ndim == x.ndim == 1
    # D[i,j] = V[i] - V[j + 1]
    D = v[:-1, None] - v[None, 1:]
    # cross(x)[i, j] = sum(x[i:j+1])
    D = D + cross(x)
    return torch.triu(D)


def smallest_power2_higher_or_equal_x(x):
    n = 1
    while n <= x:
        n *= 2
    return n


class SoftDQN(GFNAlgorithm):
    """SoftDQN or MunchausenDQN (implemented by Timofei Gritsaev tgritsaev). Implements
    - Generative Flow Networks as Entropy-Regularized RL Daniil Tiapkin, Nikita Morozov, Alexey Naumov, Dmitry Vetrov
    https://arxiv.org/abs/2310.12934

    Most code is based on the TrajectoryBalance implementation.
    """

    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        """Instanciate a DQN algorithm.

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        cfg: Config
            Hyperparameters

        update_target_every: int
            The frequency of a target_model's weights update. If equal 0, weights update every step using the `tau`.
        tau: float
            The averaging of target_model's weights parameter. Used only if `update_target_every` is equal 0
            target_model_weights = `tau` * target_model_weights + (1 - tau) * model_weights.
        is_dueling: bool
            Whether to use dueling architecture.
        munchausen_alpha: float
            The Munchausen-DQN alpha. If equal 0 used SoftDQN.
        munchausen_l0: float
            The Munchausen-DQN l0. Used only if `munchausen_alpha` is not equal 0.
        """
        self.ctx = ctx
        self.env = env
        self.global_cfg = cfg
        self.cfg = cfg.algo.dqn
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.loss_fn = self.cfg.loss_fn
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward

        self.do_parameterize_p_b = cfg.algo.do_parameterize_p_b
        self.do_predict_n = cfg.algo.do_predict_n

        # new parameters
        # the same architecture as the model
        self.target_model = GraphTransformerGFN(
            ctx,
            cfg,
            do_bck=self.do_parameterize_p_b,
            num_graph_out=self.do_predict_n + 1,
        ).to(cfg.device)
        self.target_model.eval()
        self.update_target_every = self.cfg.update_target_every
        self.tau = self.cfg.tau
        self.is_dueling = self.cfg.is_dueling
        self.munchausen_alpha = self.cfg.munchausen_alpha
        self.munchausen_l0 = self.cfg.munchausen_l0
        self.entropy_coef = 1 / (1 - self.munchausen_alpha)

        if self.cfg.rb_size:
            self.rb = []
            # PrioritizedReplayBuffer(
            #     alpha=self.cfg.per_alpha,
            #     beta=self.cfg.per_beta,
            #     storage=ListStorage(max_size=self.cfg.rb_size),
            #     collate_fn=lambda x: x,  # To store molecule data
            # )

        # Munchausen-DQN sampling is depended on the `entropy_coef = 1 / (1 - munchausen_alpha)`
        self.sample_temp = self.entropy_coef
        self.graph_sampler = GraphSampler(
            ctx,
            env,
            self.max_len,
            self.max_nodes,
            self.sample_temp,
            pad_with_terminal_state=self.do_parameterize_p_b,
        )

        self.iter_n = 0

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: nn.Module
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)
        return data

    def create_training_data_from_graphs(
        self,
        graphs,
        model: Optional[nn.Module] = None,
        cond_info: Optional[Tensor] = None,
        random_action_prob: Optional[float] = None,
    ):
        """Generate trajectories from known endpoints
        (copied and pasted from the trajectory_balance.py with a minor change).

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: nn.Module
           The model used for backward sampling
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        if self.cfg.do_sample_p_b:
            assert model is not None and cond_info is not None and random_action_prob is not None
            dev = get_worker_device()
            cond_info = cond_info.to(dev)
            return self.graph_sampler.sample_backward_from_graphs(
                graphs, model if self.do_parameterize_p_b else None, cond_info, random_action_prob
            )
        trajs: List[Dict[str, Any]] = [{"traj": generate_forward_trajectory(i)} for i in graphs]
        for traj in trajs:
            n_back = [
                self.env.count_backward_transitions(gp, check_idempotent=self.cfg.do_correct_idempotent)
                for gp, _ in traj["traj"][1:]
            ] + [1]
            traj["bck_logprobs"] = (1 / torch.tensor(n_back).float()).log().to(get_worker_device())
            traj["result"] = traj["traj"][-1][0]
            if self.do_parameterize_p_b:
                traj["bck_a"] = [GraphAction(GraphActionType.Stop)] + [self.env.reverse(g, a) for g, a in traj["traj"]]
                # There needs to be an additonal node when we're parameterizing P_B,
                # See sampling with parametrized P_B
                traj["traj"].append(copy.deepcopy(traj["traj"][-1]))
                traj["is_sink"] = [0 for _ in traj["traj"]]
                traj["is_sink"][-1] = 1
                traj["is_sink"][-2] = 1
                assert len(traj["bck_a"]) == len(traj["traj"]) == len(traj["is_sink"])
        return trajs

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information,
        copied and pasted from the trajectory_balance.py with some deletions.

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
        actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        if self.do_parameterize_p_b:
            batch.bck_actions = torch.tensor(
                [
                    self.ctx.GraphAction_to_ActionIndex(g, a)
                    for g, a in zip(torch_graphs, [i for tj in trajs for i in tj["bck_a"]])
                ]
            )
            batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        # compute_batch_losses expects these two optional values, if someone else doesn't fill them in, default to 0
        batch.num_offline = 0
        batch.num_online = 0

        return batch

    def _loss(self, x):
        if self.loss_fn == LossFN.MSE:
            return x * x
        elif self.loss_fn == LossFN.MAE:
            return torch.abs(x)
        elif self.loss_fn == LossFN.HUB:
            ax = torch.abs(x)
            return torch.where(ax < 1, 0.5 * ax * ax, ax - 0.5)
        elif self.loss_fn == LossFN.GHL:
            ax = self.cfg.loss_fn_par * x
            return torch.logaddexp(ax, -ax) - np.log(2)
        else:
            raise NotImplementedError()

    def _n_loss(self, method, P_N, N):
        n = len(N)
        if method == NLoss.SubTB1:
            return self._loss(subTB(N, -P_N)).sum() / (n * n - n) * 2
        elif method == NLoss.TermTB1:
            return self._loss(subTB(N, -P_N)[:, 0]).mean()
        elif method == NLoss.StartTB1:
            # return self._loss(subTB(N, -P_N)[0, :]).mean()
            return self._loss(N[1:] + torch.cumsum(P_N, -1)).mean()
        elif method == NLoss.TB:
            return self._loss(P_N.sum() + N[-1])
        elif method == NLoss.Transition:
            return self._loss(N[1:] + P_N - N[:-1]).mean()
        else:
            raise NotImplementedError()

    def n_loss(self, P_N, N, traj_lengths):
        dev = traj_lengths.device
        num_trajs = len(traj_lengths)
        total_loss = torch.zeros(num_trajs, device=dev)
        if self.cfg.n_loss == NLoss.none:
            return total_loss
        assert self.do_parameterize_p_b

        x = torch.cumsum(traj_lengths, 0)
        for ep, (s_idx, e_idx) in enumerate(zip(shift_right(x), x)):
            # the last state is the same as the first state
            e_idx -= 1
            total_loss[ep] = self._n_loss(self.cfg.n_loss, P_N[s_idx : e_idx - 1], N[s_idx:e_idx])
        return total_loss

    def compute_batch_losses(
        self,
        model: nn.Module,
        sampled_batch: gd.Batch,
        it: Optional[int] = None,
        num_bootstrap: int = 0,  # type: ignore[override],
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: nn.Module
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info).
        batch: gd.Batch
            batch of graphs inputs as per constructed by `self.construct_batch`.
        it: Optional[int]
            The number of iteration, which is used for updating the target_model's weights.
        num_bootstrap: int
            the number of trajectories for which the reward loss is computed. Ignored if 0."""

        device = sampled_batch.x.device
        sampled_num = sampled_batch.num_graphs
        replayed_num = 0
        replayed_indexes, trash_indexes = [], []
        num_trajs = int(sampled_batch.traj_lens.shape[0])
        batch_idx = torch.arange(num_trajs, device=device).repeat_interleave(sampled_batch.traj_lens)
        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        cond_info = getattr(sampled_batch, "cond_info", None)

        # The position of the last graph of each trajectory
        traj_cumlen = torch.cumsum(sampled_batch.traj_lens, 0)
        final_graph_idx = traj_cumlen - 1
        # The position of the first graph of each trajectory
        first_graph_idx = shift_right(traj_cumlen)
        final_graph_idx_1 = torch.maximum(final_graph_idx - 1, first_graph_idx)

        new_batch_size = smallest_power2_higher_or_equal_x(sampled_num)
        replayed_mols = []
        replayed_log_rewards = []
        replayed_is_final_graph = []
        replayed_cond_info = []
        replayed_actions = []
        replayed_bck_actions = []
        replayed_is_sink = []
        replayed_log_p_B = []
        print(48 * "-" + f" iter={self.iter_n}", flush=True)
        self.iter_n += 1
        # print(f"{model(sampled_batch, cond_info[batch_idx])=}")
        while len(self.rb) and sampled_num + len(replayed_mols) + 1 < new_batch_size:
            replayed_transitions: List[Transition]

            new_transitions_num = (new_batch_size - sampled_num - len(replayed_mols) - 1) // 2
            # replayed_transitions, replayed_info = self.rb.sample(new_transitions_num, True)
            replayed_transitions = self.rb[:2]
            for i, trans in enumerate(replayed_transitions):
                # @dataclass
                # class Transition:
                #     mol: gd.Data
                #     next_mol: gd.Data
                #     is_final_graph: bool
                #     log_reward: Optional[float]
                #     cond_info: torch.Tensor
                #     next_cond_info: torch.Tensor
                #     actions: ActionIndex
                #     next_bck_actions: ActionIndex
                #     is_sink: bool
                #     next_log_p_B: Optional[torch.Tensor]
                replayed_indexes.append(sampled_num + len(replayed_mols))
                replayed_num += 1
                replayed_mols.append(trans.mol.to(device))
                if trans.log_reward:
                    replayed_log_rewards.append(trans.log_reward)
                replayed_cond_info.append(trans.cond_info)
                replayed_actions.append(trans.actions)
                replayed_bck_actions.append(trans.bck_actions)
                replayed_is_sink.append(trans.is_sink)
                if not self.do_parameterize_p_b:
                    replayed_log_p_B.append(trans.log_p_B)
                replayed_is_final_graph.append(trans.is_final_graph)

                # print(trans.is_final_graph)
                if not trans.is_final_graph:
                    trash_indexes.append(sampled_num + len(replayed_mols))
                    replayed_mols.append(trans.next_mol.to(device))
                    replayed_cond_info.append(trans.next_cond_info)
                    replayed_actions.append(torch.zeros_like(trans.actions))
                    replayed_bck_actions.append(trans.next_bck_actions)
                    replayed_is_sink.append(torch.nan)
                    if not self.do_parameterize_p_b:
                        replayed_log_p_B.append(trans.next_log_p_B)
                    replayed_is_final_graph.append(False)

                # print(16 * "-", flush=True)
                # print(f"iter: {i}, from {trans.idx} and used {trans.used}")
                # trans.used = True
                # print(f"{self.ctx.collate(replayed_mols)=}")
                # print(f"{torch.stack(replayed_actions).to(device=device)=}")
                # print(
                #     model(self.ctx.collate(replayed_mols), torch.stack(replayed_cond_info).to(device))[0].log_prob(
                #         torch.stack(replayed_actions).to(device=device)
                #     ),
                #     flush=True,
                # )
                # print(16 * "-")

        replayed_mols = copy.deepcopy(replayed_mols)
        print(f"{sampled_num=}, {replayed_num=}")

        print(f"{sampled_batch.to_data_list()=}")
        print(f"{replayed_mols=}")
        full_batch = self.ctx.collate(sampled_batch.to_data_list() + replayed_mols)

        log_rewards = (
            torch.cat([sampled_batch.log_rewards, torch.tensor(replayed_log_rewards, device=device)])
            if replayed_num
            else sampled_batch.log_rewards
        )
        assert log_rewards.ndim == 1
        clip_log_R = torch.maximum(
            log_rewards, torch.tensor(self.global_cfg.algo.illegal_action_logreward, device=device)
        ).float()

        batched_cond_info = (
            torch.cat([cond_info[batch_idx], torch.stack(replayed_cond_info).to(device)])
            if replayed_num
            else cond_info[batch_idx]
        )
        actions = (
            torch.cat([sampled_batch.actions, torch.stack(replayed_actions).to(device)])
            if replayed_num
            else sampled_batch.actions
        )
        bck_actions = (
            torch.cat([sampled_batch.bck_actions, torch.stack(replayed_bck_actions).to(device)])
            if replayed_num
            else sampled_batch.bck_actions
        )
        # print(f"{bck_actions=}")
        is_sink = (
            torch.cat([sampled_batch.is_sink, torch.tensor(replayed_is_sink).to(device=device)])
            if replayed_num
            else sampled_batch.is_sink
        )
        # invalid_mask = 1 - torch.cat([sampled_batch.is_valid, torch.tensor(replayed_is_valid)])
        if not self.do_parameterize_p_b:
            log_p_B = (
                torch.cat([sampled_batch.log_p_B, torch.tensor(replayed_log_p_B, device=device)])
                if replayed_num
                else sampled_batch.log_p_B
            )

        is_final_graph = torch.zeros(sampled_num + len(replayed_mols), device=device)
        is_final_graph[final_graph_idx] = 1
        if replayed_num:
            is_final_graph[sampled_num:] = torch.tensor(replayed_is_final_graph, device=device)
        is_final_graph = is_final_graph.to(torch.bool)

        # Forward pass of the model, returns a GraphActionCategorical representing the forward
        # policy P_F, optionally a backward policy P_B, and per-graph outputs (e.g. F(s) in SubTB).
        if self.do_parameterize_p_b:
            print(f"{full_batch=}")
            print(f"model.forward(...) call")
            fwd_cat, bck_cat, per_graph_out = model(full_batch, batched_cond_info)
            print(f"model.forward(...) is done.")
            print("in if self.do_parameterize_p_b:")
            print(f"{fwd_cat=}")
            print(f"{bck_cat=}")
            print(f"{per_graph_out=}")
            print(f"target_model.forward(...) call")
            target_fwd_cat, target_bck_cat, target_per_graph_out = self.target_model(full_batch, batched_cond_info)
            print(f"target_model.forward(...) is done.")
        else:
            fwd_cat, per_graph_out = model(full_batch, batched_cond_info)
            target_fwd_cat, target_per_graph_out = self.target_model(full_batch, batched_cond_info)

        # Compute the log prob of each action in the trajector
        # just naively take the logprob of the actions we took
        log_p_F = fwd_cat.log_prob(actions)
        print(f"{actions[sampled_num:]=}")
        print(f"{log_p_F[sampled_num:]=}")
        if self.do_parameterize_p_b:
            log_p_B = bck_cat.log_prob(bck_actions)
            print(f"{log_p_B=}")
            print(f"{bck_cat=}")
            target_log_p_B = target_bck_cat.log_prob(bck_actions)
        if self.do_parameterize_p_b:
            # If we're modeling P_B then trajectories are padded with a virtual terminal state sF,
            # zero-out the logP_F of those states
            log_p_F[is_final_graph] = 0

            # Force the pad states' F(s) prediction to be R

            per_graph_out[is_final_graph, 0] = clip_log_R

            # To get the correct P_B we need to shift all predictions by 1 state, and ignore the
            # first P_B prediction of every trajectory.
            # Our batch looks like this:
            # [(s1, a1), (s2, a2), ..., (st, at), (sF, None),   (s1, a1), ...]
            #                                                   ^ new trajectory begins
            # For the P_B of s1, we need the output of the model at s2.

            # We also have access to the is_sink attribute, which tells us when P_B must = 1, which
            # we'll use to ignore the last padding state(s) of each trajectory. This by the same
            # occasion masks out the first P_B of the "next" trajectory that we've shifted.
            target_log_p_B = torch.roll(target_log_p_B, -1, 0) * (1 - is_sink)
            log_p_B = torch.roll(log_p_B, -1, 0) * (1 - is_sink)

            target_log_p_B[torch.isnan(target_log_p_B)] = 0
            log_p_B[torch.isnan(log_p_B)] = 0
        else:
            target_log_p_B = log_p_B
        assert log_p_F.shape == target_log_p_B.shape

        if self.do_predict_n:
            log_n_preds = per_graph_out[:sampled_num, 1]
            log_n_preds[first_graph_idx] = 0
        else:
            log_n_preds = None

        if self.cfg.n_loss == NLoss.TB:
            log_traj_n = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
            n_loss = self._loss(log_traj_n + log_n_preds[final_graph_idx_1])
        else:
            n_loss = self.n_loss(log_p_B, log_n_preds, sampled_batch.traj_lens)

        if self.ctx.has_n() and self.do_predict_n:
            analytical_maxent_backward = self.analytical_maxent_backward(sampled_batch, first_graph_idx)
            if self.do_parameterize_p_b:
                analytical_maxent_backward = torch.roll(analytical_maxent_backward, -1, 0) * (1 - is_sink)
        else:
            analytical_maxent_backward = None

        if self.cfg.backward_policy in [Backward.GSQL, Backward.GSQLA]:
            log_p_B = torch.zeros_like(log_p_B)
            nzf = torch.maximum(first_graph_idx, final_graph_idx - 1)
            if self.cfg.backward_policy == Backward.GSQLA:
                log_p_B[nzf] = -sampled_batch.log_n
            else:
                log_p_B[nzf] = -log_n_preds[nzf]
                # this is due to the fact that n(s_0)/n(s1) * n(s1)/ n(s2) = n(s_0)/n(s2) = 1 / n(s)
            # this is not final_graph_idx because we throw away the last thing
        elif self.cfg.backward_policy == Backward.MaxentA:
            log_p_B = analytical_maxent_backward

        if self.do_parameterize_p_b:
            # Life is pain, log_p_B is one unit too short for all trajs

            log_p_B_unif = torch.zeros_like(log_p_B)
            for i, (s, e) in enumerate(zip(first_graph_idx, traj_cumlen)):
                log_p_B_unif[s : e - 1] = log_p_B[s - i : e - 1 - i]

            if self.cfg.backward_policy == Backward.Uniform:
                log_p_B = log_p_B_unif
        else:
            log_p_B_unif = log_p_B

        if self.cfg.backward_policy in [Backward.Maxent, Backward.GSQL]:
            log_p_B = log_p_B.detach()

        print(f"{log_p_F[sampled_num:]=}")
        print(f"{per_graph_out[sampled_num:, 0]=}")
        if self.is_dueling:
            q_states = log_p_F + per_graph_out[:, 0]
            F_sm = target_per_graph_out[:, 0].roll(-1)
        else:
            # see the equation 12 in the https://arxiv.org/pdf/2310.12934
            q_states = log_p_F
            F_sm = self.entropy_coef * target_fwd_cat.logsumexp(temperature=self.entropy_coef).roll(-1)
        F_sm[is_final_graph] = clip_log_R
        target = F_sm + target_log_p_B
        print(f"{sum(is_final_graph[sampled_num:])=}, {clip_log_R=}")
        print(f"{target_log_p_B[sampled_num:]=}")
        print(f"{q_states[sampled_num:]=}")
        print(f"{target[sampled_num:]=}")

        if self.munchausen_alpha > 0:
            target_log_p_F = target_fwd_cat.log_prob(actions)
            munchausen_penalty = torch.clamp(self.entropy_coef * target_log_p_F, min=self.munchausen_l0)
            target += self.munchausen_alpha * munchausen_penalty

        transition_losses = self._loss(q_states - target)

        print(f"{transition_losses=}")
        print(f"{trash_indexes=}")
        transition_losses[trash_indexes] = 0.0
        transition_losses[is_sink.to(torch.bool)] *= self.cfg.leaf_coef
        print(f"{transition_losses=}")

        # This is the log probability of each trajectory
        traj_log_p_F = scatter(log_p_F[:sampled_num], batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_unif_log_p_B = scatter(log_p_B_unif[:sampled_num], batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B[:sampled_num], batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        n_loss = n_loss.mean()
        print("!", f"{transition_losses=}")
        loss = transition_losses.sum() / (sampled_num + replayed_num)
        # + self.cfg.n_loss_multiplier * n_loss
        print(f"!!! {loss=}")
        info = {
            # "offline_loss": traj_losses[: sampled_batch.num_offline].mean() if sampled_batch.num_offline > 0 else 0,
            # "online_loss": traj_losses[sampled_batch.num_offline :].mean() if sampled_batch.num_online > 0 else 0,
            # "invalid_trajectories": (
            #     invalid_mask.sum() / sampled_batch.num_online if sampled_batch.num_online > 0 else 0
            # ),
            # "invalid_logprob": (invalid_mask * traj_log_p_F[:sampled_num]).sum() / (invalid_mask.sum() + 1e-4),
            # "invalid_losses": (invalid_mask * traj_losses[:sampled_num]).sum() / (invalid_mask.sum() + 1e-4),
            "backward_vs_unif": (traj_unif_log_p_B - traj_log_p_B).pow(2).mean(),
            "loss": loss.item(),
            "n_loss": n_loss,
            "tlm": (
                torch.tensor([0.0], requires_grad=True)
                if torch.count_nonzero(log_p_B) == 0
                else -torch.sum(log_p_B) / torch.count_nonzero(log_p_B)
            ),
            "batch_entropy": -traj_log_p_F.mean(),
            "traj_lens": sampled_batch.traj_lens.float().mean(),
            "replayed_num": replayed_num,
        }
        if self.ctx.has_n() and self.do_predict_n:
            info["n_loss_pred"] = scatter(
                (log_n_preds - sampled_batch.log_ns) ** 2, batch_idx, dim=0, dim_size=num_trajs, reduce="sum"
            ).mean()
            info["n_final_loss"] = torch.mean((log_n_preds[final_graph_idx] - sampled_batch.log_n) ** 2)
            if self.do_parameterize_p_b:
                info["n_loss_tgsql"] = torch.mean((-sampled_batch.log_n - traj_log_p_B) ** 2)
                d = analytical_maxent_backward - log_p_B
                d = d * d
                d[final_graph_idx] = 0
                info["n_loss_maxent"] = scatter(d, batch_idx, dim=0, dim_size=num_trajs, reduce="sum").mean()

        if model.training:
            # `it` could be None only if `update_target_every` == 0
            with torch.no_grad():
                if self.update_target_every > 0 and it % self.update_target_every == 0:
                    for _a, b in zip(model.parameters(), self.target_model.parameters()):
                        b.data = _a.data
                    # usually people use: self.target_model.load_state_dict(model.state_dict()), but it fails
                elif self.tau > 0:
                    for _a, b in zip(model.parameters(), self.target_model.parameters()):
                        b.data.mul_(self.tau).add_((1 - self.tau) * _a)

            # print(transition_losses)
            # if replayed_num:
            #     assert replayed_info["index"].shape[0] == len(
            #         replayed_indexes
            #     ), f"{replayed_indexes.shape=}, {len(replayed_indexes)=}"
            #     print(f"{replayed_info['index']=}, {transition_losses[replayed_indexes]=}")
            #     self.rb.update_priority(replayed_info["index"], transition_losses[replayedf_indexes])

            to_rb = []
            num_final_graph = 0

            sampled_batch_cpu = detach_and_cpu(copy.deepcopy(sampled_batch))
            log_rewards = log_rewards
            is_final_graph = is_final_graph.to("cpu")
            batched_cond_info = batched_cond_info.to("cpu")
            actions = actions.to("cpu")
            bck_actions = bck_actions.to("cpu")
            is_sink = is_sink.to("cpu")
            if not self.do_parameterize_p_b:
                log_p_B_on_cpu = sampled_batch.log_p_B.to("cpu")

            # print(f"ADDING {sampled_num=} {len(sampled_batch_cpu)=}")
            for i in range(sampled_num):
                if is_final_graph[i]:
                    log_reward = log_rewards[num_final_graph]
                    num_final_graph += 1
                else:
                    log_reward = None
                to_rb.append(
                    Transition(
                        sampled_batch_cpu[i],
                        sampled_batch_cpu[i + 1] if not is_final_graph[i] else None,
                        is_final_graph[i],
                        log_reward,
                        batched_cond_info[i],
                        batched_cond_info[i + 1] if not is_final_graph[i] else None,
                        actions[i],
                        bck_actions[i],
                        bck_actions[i + 1] if not is_final_graph[i] else torch.zeros_like(bck_actions[i]),
                        is_sink[i],
                        log_p_B_on_cpu[i] if (not self.do_parameterize_p_b and not is_final_graph[i]) else None,
                        log_p_B_on_cpu[i + 1] if (not self.do_parameterize_p_b and not is_final_graph[i]) else None,
                        i,
                    )
                )
            assert num_final_graph == log_rewards.shape[0]

            new_buffer_indexes = self.rb.extend(to_rb)
            # assert new_buffer_indexes.shape == transition_losses[:sampled_num].shape
            # self.rb.update_priority(new_buffer_indexes, transition_losses[:sampled_num])

        return loss, info

    def analytical_maxent_backward(self, batch, first_graph_idx):
        s = shift_right(batch.log_ns)
        s[first_graph_idx] = 0
        return s - batch.log_ns
