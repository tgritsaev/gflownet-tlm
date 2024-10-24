from copy import deepcopy

import torch
from gfn.modules import DiscretePolicyEstimator, GFNModule
from torchtyping import TensorType as TT

from gfn.containers import Trajectories, Transitions
from gfn.gflownet import GFlowNet
from gfn.states import States
from gfn.samplers import Sampler


class RescaledDPE(DiscretePolicyEstimator):
    def __init__(self, module: DiscretePolicyEstimator, scaling: float = 1.0):
        super().__init__(
            module.env,
            module.module,
            module._forward,
            module._greedy_eps,
            module.temperature,
            module.sf_bias,
            module.epsilon,
        )
        self.scaling = scaling

    def forward(self, states: States):
        return super().forward(states) / self.scaling


class SoftDQNGFlowNet(GFlowNet):
    def __init__(
        self,
        q: GFNModule,
        q_target: GFNModule,
        pb: GFNModule,
        smooth_pb: bool = True,
        on_policy: bool = False,
        is_double: bool = False,
        entropy_coeff: float = 1.0,
        munchausen_alpha: float = 0.0,
        munchausen_l0: float = -2.0,
    ):
        super().__init__()
        self.q = q
        self.pb = pb
        self.on_policy = on_policy
        self.is_double = is_double

        self.q_target = q_target
        self.q_target.load_state_dict(self.q.state_dict())
        self.smooth_pb = smooth_pb
        if smooth_pb:
            self.pb_target = deepcopy(pb)
            self.pb_target.load_state_dict(self.pb.state_dict())

        self.entropy_coeff = entropy_coeff
        self.munchausen_alpha = munchausen_alpha
        self.munchausen_l0 = munchausen_l0

        self.max_n_samples = 20000

    def sample_trajectories(self, n_samples: int = 1000) -> Trajectories:
        sampler = Sampler(estimator=RescaledDPE(self.q, scaling=self.entropy_coeff))
        trajectories = sampler.sample_trajectories(n_trajectories=n_samples)
        return trajectories

    def sample_terminating_states(self, n_samples: int) -> States:
        """Rolls out the parametrization's policy and returns the terminating states.

        Args:
            n_samples: number of terminating states to be sampled.
        Returns:
            States: sampled terminating states object.
        """
        n_remained_samples = n_samples
        terminating_states = None
        while n_remained_samples > 0:
            current_n_samples = min(n_remained_samples, self.max_n_samples)
            sampler = Sampler(estimator=RescaledDPE(self.q, scaling=self.entropy_coeff))

            is_sampled_successfull = False
            while not is_sampled_successfull:
                try:
                    current_terminating_states = sampler.sample_trajectories(
                        n_trajectories=current_n_samples
                    ).last_states
                    is_sampled_successfull = True
                except Exception as exception:
                    print(exception)

            if terminating_states is None:
                terminating_states = current_terminating_states
            else:
                terminating_states.extend(current_terminating_states)
            n_remained_samples -= current_n_samples
        return terminating_states

    def update_target_nets(self, tau=0.05):
        if tau == 1.0:
            self.q_target.load_state_dict(self.q.state_dict())
        else:

            def update_params(net, target_net):
                for param, target_param in zip(net.parameters(), target_net.parameters()):
                    target_param.data.mul_(1 - tau)
                    torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)

            with torch.no_grad():
                update_params(self.q, self.q_target)
                if self.smooth_pb:
                    update_params(self.pb, self.pb_target)

    def get_scores(self, transitions: Transitions):
        """Given a batch of transitions, calculate the scores.

        Args:
            transitions: a batch of transitions.

        Raises:
            ValueError: when supplied with backward transitions.
            AssertionError: when log rewards of transitions are None.
        """
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions.is_dummy)

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        q_s = self.q(states)
        q_s[~states.forward_masks] = -float("inf")

        preds = torch.gather(q_s, 1, actions.tensor).squeeze(-1)
        targets = torch.zeros_like(preds)

        valid_next_states = transitions.next_states[~transitions.is_done]
        non_exit_actions = actions[~actions.is_exit]

        if self.smooth_pb:
            pb_model = self.pb_target
        else:
            pb_model = self.pb
        module_output = pb_model(valid_next_states)
        valid_log_pb_actions = pb_model.to_probability_distribution(valid_next_states, module_output).log_prob(non_exit_actions.tensor)

        valid_transitions_is_done = transitions.is_done[~transitions.states.is_sink_state]

        with torch.no_grad():
            q_sn_target = self.q_target(valid_next_states)
            q_sn_target[~valid_next_states.forward_masks] = -float("inf")
            if not self.is_double:
                valid_v_target_next = self.entropy_coeff * torch.logsumexp(
                    q_sn_target / self.entropy_coeff, dim=-1
                ).squeeze(-1)
            else:
                q_sn = self.q(valid_next_states) / self.entropy_coeff
                q_sn[~valid_next_states.forward_masks] = -float("inf")
                policy_sn = torch.exp(q_sn)
                policy_sn /= policy_sn.sum(dim=-1, keepdim=True)
                log_policy_sn = torch.log(policy_sn + 1e-9)
                q_sn_target[~valid_next_states.forward_masks] = 0
                valid_v_target_next = torch.sum(
                    policy_sn * (q_sn_target - self.entropy_coeff * log_policy_sn), dim=-1
                ).squeeze(-1)

        targets[~valid_transitions_is_done] = valid_log_pb_actions + valid_v_target_next
        assert transitions.log_rewards is not None
        valid_transitions_log_rewards = transitions.log_rewards[~transitions.states.is_sink_state]
        targets[valid_transitions_is_done] = valid_transitions_log_rewards[valid_transitions_is_done]

        if self.munchausen_alpha > 0:
            with torch.no_grad():
                q_s_target = self.q_target(states) / self.entropy_coeff
                q_s_target[~states.forward_masks] = -float("inf")
                v_s_target = torch.logsumexp(q_s_target, dim=-1).squeeze(-1)
                q_sa_target = torch.gather(q_s_target, 1, actions.tensor).squeeze(-1)
                log_pf_target = q_sa_target - v_s_target
                penalty = (self.entropy_coeff * log_pf_target).clamp(min=self.munchausen_l0, max=0.0)
            targets += self.munchausen_alpha * penalty

        scores = preds - targets

        return scores

    def loss(self, transitions: Transitions) -> TT[0, float]:
        _, _, scores = self.get_scores(transitions)
        loss = torch.mean(scores**2)

        if torch.isnan(loss):
            raise ValueError("loss is nan")

        return loss

    def get_pb_loss(self, transitions: Transitions):
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        states = transitions.states
        actions = transitions.actions

        if states.batch_shape != tuple(actions.batch_shape):
            raise ValueError("Something wrong happening with log_pf evaluations")

        valid_next_states = transitions.next_states[~transitions.is_done]
        module_output = self.pb(valid_next_states)
        log_pb = self.pb.to_probability_distribution(valid_next_states, module_output).log_prob(
            actions.tensor[~transitions.is_done]
        )

        return -torch.sum(log_pb)

    def to_training_samples(self, trajectories: Trajectories) -> Transitions:
        return trajectories.to_transitions()
