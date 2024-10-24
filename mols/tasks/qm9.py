import argparse
import datetime
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from mols.algo.config import Backward, NLoss, TBVariant
import mols.models.mxmnet as mxmnet
from mols import GFNTask, LogScalar, ObjectProperties
from mols.config import Config, init_empty
from mols.data.qm9 import QM9Dataset
from mols.envs.mol_building_env import MolBuildingEnvContext
from mols.online_trainer import StandardOnlineTrainer
from mols.utils.conditioning import TemperatureConditional
from mols.utils.misc import get_worker_device
from mols.utils.transforms import to_logreward
from mols.utils.utils import set_all_random


parser = argparse.ArgumentParser()

parser.add_argument("--debug", action="store_true")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--temp", type=int, default=10)
parser.add_argument("--n", type=int, default=20_000)

parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--lrd", type=int, default=20_000)
parser.add_argument("--blr", type=float)
parser.add_argument("--blrd", type=int, default=20_000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--train_policy", type=str, choices=["tb", "db", "subtb", "dqn"], default="tb")
parser.add_argument("--backward_approach", type=str, choices=["uniform", "naive", "tlm", "maxent"], default="uniform")

parser.add_argument("--reverse_updates_order", action="store_true")
parser.add_argument("--no_sampling_model", action="store_true")
parser.add_argument("--one_forward_step", action="store_true")

parser.add_argument("--rp", type=float, default=0.05)
parser.add_argument("--ap", type=str, choices=["", "linear", "exp"], default="linear")
parser.add_argument("--aexp", type=float, default=0.2)

# Munchausen DQN hyperparameters
parser.add_argument("--not_dueling", action="store_true")
parser.add_argument("--m_alpha", type=float, default=0.15)
parser.add_argument("--m_l0", type=float, default=-500)

parser.add_argument("--rand_pb_init", action="store_true")

args = parser.parse_args()


class QM9GapTask(GFNTask):
    """This class captures conditional information generation and reward transforms"""

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.device = get_worker_device()
        self.models = self.load_task_models(cfg.task.qm9.model_path)
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        # TODO: fix interface
        self._min, self._max, self._percentile_95 = self.dataset.get_stats("gap", percentile=0.05)  # type: ignore
        self._width = self._max - self._min
        self._rtrans = "unit+95p"  # TODO: hyperparameter

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
        """Transforms a target quantity y (e.g. the LUMO energy in QM9) to a positive reward scalar"""
        if self._rtrans == "exp":
            flat_r = np.exp(-(y - self._min) / self._width)
        elif self._rtrans == "unit":
            flat_r = 1 - (y - self._min) / self._width
        elif self._rtrans == "unit+95p":
            # Add constant such that 5% of rewards are > 1
            flat_r = 1 - (y - self._percentile_95) / self._width
        else:
            raise ValueError(self._rtrans)
        return ObjectProperties(flat_r)

    def inverse_reward_transform(self, rp):
        if self._rtrans == "exp":
            return -np.log(rp) * self._width + self._min
        elif self._rtrans == "unit":
            return (1 - rp) * self._width + self._min
        elif self._rtrans == "unit+95p":
            return (1 - rp + (1 - self._percentile_95)) * self._width + self._min

    def load_task_models(self, path):
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        # TODO: this path should be part of the config?
        try:
            state_dict = torch.load(path, map_location=self.device)
        except Exception as e:
            print(
                "Could not load model.",
                e,
                "\nModel weights can be found at",
                "https://storage.googleapis.com/emmanuel-data/models/mxmnet_gap_model.pt",
            )
        gap_model.load_state_dict(state_dict)
        gap_model.to(self.device)
        gap_model = self._wrap_model(gap_model)
        return {"mxmnet_gap": gap_model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, graphs: List[gd.Data]) -> Tensor:
        graphs_list = [i for i in graphs if i is not None]
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(
            self.models["mxmnet_gap"].device if hasattr(self.models["mxmnet_gap"], "device") else get_worker_device()
        )
        preds = self.models["mxmnet_gap"](batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV  # type: ignore[attr-defined]
        preds[preds.isnan()] = 1
        preds = self.reward_transform(preds).clip(1e-4, 2).reshape(-1)
        return preds

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [mxmnet.mol2graph(i) for i in mols]  # type: ignore[attr-defined]

        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            # all elements are False
            return ObjectProperties(torch.zeros((is_valid.shape[0], 1))), is_valid
        if (~is_valid).any():
            # at least 1 element is False
            # print("!!!!")
            i = 0
            for j in range(len(graphs)):
                # print(f"{j} out of {i}")
                if graphs[j] is not None:
                    # print(f"{j} GOOD")
                    i = j
                    break
            for j in range(len(graphs)):
                if graphs[j] is None:
                    # print(f"fix {j}")
                    graphs[j] = graphs[i]
            is_valid = torch.tensor([i is not None for i in graphs]).bool()
            # print(is_valid)

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        if len(preds) != is_valid.sum():
            return ObjectProperties(torch.zeros((is_valid.shape[0], 1))), is_valid
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


class QM9GapTrainer(StandardOnlineTrainer):
    def set_default_hps(self, cfg: Config):
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.overwrite_existing_exp = True
        cfg.task.qm9.h5_path = "mols/data/qm9.h5"
        cfg.task.qm9.model_path = "mols/data/mxmnet_gap_model.pt"

        if args.debug:
            print("DEBUG MODE!!!!!!!", flush=True)
            # cfg.device = "cpu"
            cfg.log_dir = f"./debug_qm9_logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-"
            cfg.print_every = 5
            cfg.validate_every = 10
            cfg.monte_carlo_corr_n = 2
            cfg.num_validation_gen_steps = 1
            cfg.num_training_steps = 100
        else:
            cfg.log_dir = f"./qm9_logs/seed={args.seed}-"
            cfg.print_every = 100
            cfg.validate_every = 1000
            cfg.num_validation_gen_steps = 10
            cfg.num_training_steps = args.n
        set_all_random(args.seed)
        cfg.num_final_gen_steps = 0
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.log_dir += f"steps={args.n // 1000}k-"

        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [args.temp]
        cfg.log_dir += f"temp={args.temp}-"
        cfg.cond.temperature.num_thermometer_dim = 32

        if args.train_policy in ["tb", "db", "subtb"]:
            cfg.algo.method = "TB"
            self.train_policy = args.train_policy
            cfg.log_dir += self.train_policy + "-"
            if self.train_policy == "tb":
                cfg.algo.tb.variant = TBVariant.TB
            elif self.train_policy == "db":
                cfg.algo.tb.variant = TBVariant.DB
            else:
                cfg.algo.tb.variant = TBVariant.SubTB1
            algo_method = cfg.algo.tb
        elif args.train_policy == "dqn":
            cfg.algo.method = "DQN"
            cfg.algo.dqn.is_dueling = not args.not_dueling
            cfg.algo.dqn.munchausen_alpha = args.m_alpha
            cfg.algo.dqn.munchausen_l0 = args.m_l0
            cfg.log_dir += ("dueling_" if cfg.algo.dqn.is_dueling else "") + "dqn-"
            cfg.log_dir += f"m_alpha={args.m_alpha}-m_l0={args.m_l0}-"
            algo_method = cfg.algo.dqn
        else:
            raise KeyError

        cfg.opt.learning_rate = args.lr
        cfg.opt.lr_decay = args.lrd
        cfg.log_dir += f"lr={args.lr}-d={args.lrd // 1000}k-"
        if args.blr is not None:
            cfg.opt.backward_learning_rate = args.blr
        else:
            cfg.opt.backward_learning_rate = args.lr / 10
        cfg.opt.backward_lr_decay = args.blrd
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        if args.backward_approach != "uniform":
            cfg.log_dir += f"blr={cfg.opt.backward_learning_rate}-d={args.blrd // 1000}k-"

        cfg.model.num_emb = 128
        cfg.model.num_layers = 8
        cfg.model.unif_init = not args.rand_pb_init

        cfg.algo.num_from_policy = args.batch_size
        cfg.algo.max_nodes = 8
        cfg.algo.sampling_tau = 0.95
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = args.rp
        cfg.log_dir += f"rp={args.rp}-"
        cfg.algo.train_random_action_prob_annealing_policy = args.ap
        if args.ap != "":
            cfg.log_dir += f"annpol={args.ap}-"
            if args.ap == "exp":
                cfg.log_dir += f"exp={args.aexp}-"
                cfg.algo.train_random_action_prob_exp = args.aexp
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = args.batch_size
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_correct_idempotent = False
        cfg.algo.tb.do_sample_p_b = False

        self.backward_approach = args.backward_approach
        cfg.log_dir += args.backward_approach
        self.use_sampling_model = not args.no_sampling_model
        self.one_forward_step = args.one_forward_step
        if self.backward_approach == "uniform":
            cfg.algo.do_parameterize_p_b = False
            algo_method.backward_policy = Backward.Uniform
        else:
            cfg.algo.do_parameterize_p_b = True
            # better to implement backward via backward_policy
            # but it is for the sake of simplicity implementation
            algo_method.backward_approach = self.backward_approach
            if self.backward_approach in ["naive", "tlm"]:
                algo_method.backward_policy = Backward.Free
                cfg.log_dir += "-no_sampling_model" if args.no_sampling_model else ""
                cfg.log_dir += "-one_step" if args.one_forward_step else ""
            else:
                algo_method.n_loss = NLoss.Transition
                cfg.algo.do_predict_n = True
                if self.backward_approach == "maxent":
                    algo_method.backward_policy = Backward.Maxent
                elif self.backward_approach == "gsql":
                    algo_method.backward_policy = Backward.GSQL
        cfg.log_dir += "-rinit" if args.rand_pb_init else ""
        
        self.reverse_updates_order = args.reverse_updates_order
        cfg.log_dir += "-reverse_updates_order" if args.reverse_updates_order else ""

        cfg.replay.use = False

    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ["C", "N", "F", "O"],
            expl_H_range=[0, 1, 2, 3],
            num_cond_dim=self.task.num_cond_dim,
            allow_5_valence_nitrogen=True,
        )
        # Note: we only need the allow_5_valence_nitrogen flag because of how we generate trajectories
        # from the dataset. For example, consider tue Nitrogen atom in this: C[NH+](C)C, when s=CN(C)C, if the action
        # for setting the explicit hydrogen is used before the positive charge is set, it will be considered
        # an invalid action. However, generate_forward_trajectory does not consider this implementation detail,
        # it assumes that attribute-setting will always be valid. For the molecular environment, as of writing
        # (PR #98) this edge case is the only case where the ordering in which attributes are set can matter.

    def setup_data(self):
        self.training_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=True, targets=["gap"])
        self.test_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=False, targets=["gap"])
        self.to_terminate.append(self.training_data.terminate)
        self.to_terminate.append(self.test_data.terminate)

    def setup_task(self):
        self.task = QM9GapTask(
            dataset=self.training_data,
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)
        self.test_data.setup(self.task, self.ctx)


def main():
    trial = QM9GapTrainer(init_empty(Config()))
    trial.run()


if __name__ == "__main__":
    main()
