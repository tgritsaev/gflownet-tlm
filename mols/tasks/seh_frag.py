import argparse
import datetime
import socket
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from mols import GFNTask, LogScalar, ObjectProperties
from mols.algo.config import Backward, NLoss, TBVariant
from mols.config import Config, init_empty
from mols.envs.frag_mol_env import FragMolBuildingEnvContext, Graph
from mols.models import bengio2021flow
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
parser.add_argument("--blrd", type=int, default=5_000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_policy", type=str, choices=["tb", "db", "subtb", "dqn"], default="tb")
parser.add_argument("--backward_approach", type=str, choices=["uniform", "naive", "tlm", "maxent"], default="uniform")

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


class SEHTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `mols.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        cfg: Config,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> None:
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return self.temperature_conditional.transform(cond_info, to_logreward(flat_reward))

    def compute_reward_from_graph(self, graphs: List[Data]) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device())
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / self.reward_norm
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid
        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))

        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


SOME_MOLS = []


class LittleSEHDataset(Dataset):
    """Note: this dataset isn't used by default, but turning it on showcases some features of this codebase.

    To turn on, self `cfg.algo.num_from_dataset > 0`"""

    def __init__(self, smis) -> None:
        super().__init__()
        self.props: ObjectProperties
        self.mols: List[Graph] = []
        self.smis = smis

    def setup(self, task: SEHTask, ctx: FragMolBuildingEnvContext) -> None:
        rdmols = [Chem.MolFromSmiles(i) for i in SOME_MOLS]
        self.mols = [ctx.obj_to_graph(i) for i in rdmols]
        self.props = task.compute_obj_properties(rdmols)[0]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, index):
        return self.mols[index], self.props[index]


class SEHFragTrainer(StandardOnlineTrainer):
    task: SEHTask
    training_data: LittleSEHDataset

    def set_default_hps(self, cfg: Config):

        cfg.hostname = socket.gethostname()
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.overwrite_existing_exp = True
        if args.debug:
            print("DEBUG MODE!!!!!!!", flush=True)
            # cfg.device = "cpu"
            cfg.log_dir = f"./debug_seh_logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-"
            cfg.print_every = 10
            cfg.validate_every = 10
            cfg.monte_carlo_corr_n = 2
            cfg.num_validation_gen_steps = 1
            cfg.num_training_steps = 1000
        else:
            cfg.log_dir = f"./seh_logs/seed={args.seed}-"
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
        cfg.opt.backward_learning_rate = args.lr / 10
        cfg.opt.backward_lr_decay = args.blrd
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        if args.backward_approach != "uniform":
            cfg.log_dir += f"blr={args.lr / 10}-d={args.blrd // 1000}k-"

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

        cfg.replay.use = False

    def setup_task(self):
        self.task = SEHTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def setup_data(self):
        super().setup_data()
        self.training_data = LittleSEHDataset([])

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.seh.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)


def main():
    trial = SEHFragTrainer(init_empty(Config()))
    trial.run()


if __name__ == "__main__":
    main()
