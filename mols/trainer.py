import gc
import logging
import os
import pathlib
import pickle
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch_geometric.data as gd
import wandb
from omegaconf import OmegaConf
from rdkit import RDLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import rdkit.Chem as Chem

from mols import GFNAlgorithm, GFNTask
import mols.models.mxmnet as mxmnet
from mols.data.data_source import DataSource
from mols.data.replay_buffer import ReplayBuffer
from mols.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnv, GraphBuildingEnvContext
from mols.envs.seq_building_env import SeqBatch
from mols.utils.metrics import (
    compute_num_modes,
    monte_carlo_compute_correlation_stats,
    top_k_diversity,
    get_topk,
    compute_diverse_top_k,
)
from mols.utils.misc import create_logger, set_main_process_device, set_worker_rng_seed
from mols.utils.multiprocessing_proxy import mp_object_wrapper
from mols.utils.sqlite_log import (
    SQLiteLogHook,
    read_all_results,
    read_all_results_at_least_reward,
    clear_all_data,
)
from .config import Config


class Closable(Protocol):
    def close(self):
        pass


class GFNTrainer:
    def __init__(self, config: Config, print_config=True):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        config: Config
            The hyperparameters for the trainer.
        """
        self.print_config = print_config
        self.to_terminate: List[Closable] = []
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module
        self.replay_buffer: Optional[ReplayBuffer]
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm
        
        self.reverse_updates_order = False

        # There are three sources of config values
        #   - The default values specified in individual config classes
        #   - The default values specified in the `default_hps` method, typically what is defined by a task
        #   - The values passed in the constructor, typically what is called by the user
        # The final config is obtained by merging the three sources with the following precedence:
        #   config classes < default_hps < constructor (i.e. the constructor overrides the default_hps, and so on)
        self.default_cfg: Config = Config()
        self.set_default_hps(self.default_cfg)
        assert isinstance(self.default_cfg, Config) and isinstance(
            config, Config
        )  # make sure the config is a Config object, and not the Config class itself
        self.cfg: Config = OmegaConf.merge(self.default_cfg, config)

        self.device = torch.device(self.cfg.device)
        set_main_process_device(self.device)
        # Print the loss every `self.print_every` iterations
        self.print_every = self.cfg.print_every
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.inf_loss_cnt = 0

        self.setup()

        test_mols = []
        self.test_graphs = []
        # the next code creates the test set for SEHTask for correlation computation
        if "seh" in self.cfg.log_dir:
            # better to use `isinstance(self.task, SEHTask)`, but it leads to a circular import
            with open(f"mols/binary_test_mols.pkl", "rb") as fin:
                binary_mols = pickle.load(fin)
            fails_cnt = 0
            for i, binary_mol in enumerate(binary_mols):
                try:
                    mol = Chem.Mol(binary_mol)
                    self.test_graphs.append(self.ctx.obj_to_graph(mol))
                    test_mols.append(mol)
                except Exception as exc:
                    fails_cnt += 1
                    # print(f"Failed to create a graph from the {i}-th mol with the exception: \"{exc}\".")
            print(f"Failed to deal with {fails_cnt} mols out of {len(binary_mols)}.")
            # "Failed to deal with 53 mols out of 950."
            """
            1. The test test is the same as in the https://arxiv.org/abs/2106.04399 
            Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation.
            2. It was created with the following code run 
            in the repository https://github.com/GFNOrg/gflownet/tree/master
            ----------
            import pickle
            import gzip
            from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
            test_set = pickle.load(gzip.open("data/some_mols_U_1k.pkl.gz"))
            binary_test_mols = [test_obj[1].mol.ToBinary() for test_obj in test_set]
            with open(f"binary_test_mols.pkl", "wb") as fout:
                pickle.dump(binary_test_mols, fout)
            """
        elif "qm9" in self.cfg.log_dir:
            df = pd.HDFStore(self.cfg.task.qm9.h5_path, "r")["df"]
            cnt_by_num_atoms = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
            for i in range(0, len(df)):
                mol = Chem.MolFromSmiles(df["SMILES"][i])
                num_atoms = mol.GetNumAtoms()
                if num_atoms < 3 or num_atoms > 8 or cnt_by_num_atoms[num_atoms] > 200:
                    continue
                else:
                    if mxmnet.mol2graph(mol) is not None:
                        self.test_graphs.append(self.ctx.obj_to_graph(mol))
                        test_mols.append(mol)
                        cnt_by_num_atoms[num_atoms] += 1
        test_mols_props, test_mols_is_valid = self.task.compute_obj_properties(test_mols)
        assert test_mols_is_valid.all()
        self.test_mols_log_rewards = self.task.cond_info_to_logreward(
            self.task.sample_conditional_information(len(test_mols), None),
            test_mols_props,
        )
        print(f"test set size: {len(test_mols)}")

    def set_default_hps(self, base: Config):
        raise NotImplementedError()

    def setup_env_context(self):
        raise NotImplementedError()

    def setup_task(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_algo(self):
        raise NotImplementedError()

    def setup_data(self):
        pass

    def step(self, batch: gd.Batch, train_it: int):
        raise NotImplementedError()

    def setup(self):
        if os.path.exists(self.cfg.log_dir):
            if self.cfg.overwrite_existing_exp:
                shutil.rmtree(self.cfg.log_dir)
            else:
                raise ValueError(
                    f"Log dir {self.cfg.log_dir} already exists. Set overwrite_existing_exp=True to delete it."
                )
        os.makedirs(self.cfg.log_dir)

        RDLogger.DisableLog("rdApp.*")
        set_worker_rng_seed(self.cfg.seed)
        self.env = GraphBuildingEnv()
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()

    def _wrap_for_mp(self, obj):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if self.cfg.num_workers > 0 and obj is not None:
            wrapper = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                cast_types=(gd.Batch, GraphActionCategorical, SeqBatch),
                pickle_messages=self.cfg.pickle_mp_messages,
            )
            self.to_terminate.append(wrapper.terminate)
            return wrapper.placeholder
        else:
            return obj

    def build_callbacks(self):
        return {}

    def _make_data_loader(self, src):
        return torch.utils.data.DataLoader(
            src,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_training_data_loader(self) -> DataLoader:
        # Since the model may be used by a worker in a different process, we need to wrap it.
        # See implementation_notes.md for more details.
        model = self._wrap_for_mp(self.sampling_model)
        replay_buffer = self._wrap_for_mp(self.replay_buffer)

        if self.cfg.replay.use:
            # None is fine for either value, it will be replaced by num_from_policy, but 0 is not
            assert self.cfg.replay.num_from_replay != 0, "Replay is enabled but no samples are being drawn from it"
            assert self.cfg.replay.num_new_samples != 0, "Replay is enabled but no new samples are being added to it"

        n_drawn = self.cfg.algo.num_from_policy
        n_replayed = self.cfg.replay.num_from_replay or n_drawn if self.cfg.replay.use else 0
        n_new_replay_samples = (
            self.cfg.replay.num_new_samples or n_drawn
            if (self.cfg.replay.use or self.cfg.replay.use_for_backward)
            else None
        )
        n_from_dataset = self.cfg.algo.num_from_dataset

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, replay_buffer=replay_buffer)
        if n_from_dataset:
            src.do_sample_dataset(self.training_data, n_from_dataset, backwards_model=model)
        if n_drawn:
            src.do_sample_model(model, n_drawn, n_new_replay_samples)
        if n_replayed and replay_buffer is not None:
            src.do_sample_replay(n_replayed)
        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_validation_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)
        # TODO: we're changing the default, make sure anything that is using test data is adjusted
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        n_drawn = self.cfg.algo.valid_num_from_policy
        n_from_dataset = self.cfg.algo.valid_num_from_dataset

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        if n_from_dataset:
            src.do_dataset_in_order(self.test_data, n_from_dataset, backwards_model=model)
        if n_drawn:
            assert self.cfg.num_validation_gen_steps is not None
            # TODO: might be better to change total steps to total trajectories drawn
            src.do_sample_model_n_times(model, n_drawn, num_total=self.cfg.num_validation_gen_steps * n_drawn)

        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "valid"), self.ctx))
        for hook in self.valid_sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_final_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)

        n_drawn = self.cfg.algo.num_from_policy
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        assert self.cfg.num_final_gen_steps is not None
        # TODO: might be better to change total steps to total trajectories drawn
        src.do_sample_model_n_times(model, n_drawn, num_total=self.cfg.num_final_gen_steps * n_drawn)

        if self.cfg.log_dir:
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "final"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        tick = time.time()
        self.model.train()
        try:
            # loss, info = self.algo.compute_batch_losses(self.model, batch)
            # if not torch.isfinite(loss):
            #     raise ValueError("loss is not finite")
            info = self.step(batch, train_it)
            self.algo.step()  # This also isn't used anywhere?
            if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.model.parameters()]):
                raise ValueError("parameters are not finite")
        except ValueError as e:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            # torch.save([self.model.state_dict(), batch, loss, info], open(self.cfg.log_dir + "/dump.pkl", "wb"))
            raise e

        # if step_info is not None:
        #     info.update(step_info)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["train_time"] = time.time() - tick
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        tick = time.time()
        self.model.eval()
        _, info = self.algo.compute_batch_losses(self.model, batch)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["eval_time"] = time.time() - tick
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        if logger is None:
            logger = create_logger(logfile=self.cfg.log_dir + "/train.log")
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        if self.cfg.num_final_gen_steps:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.cfg.start_at_step + 1
        num_training_steps = self.cfg.num_training_steps
        logger.info("Starting training")
        start_time = time.time()
        for it, batch in zip(range(start, 1 + num_training_steps), cycle(train_dl)):
            # the memory fragmentation or allocation keeps growing, how often should we clean up?
            # is changing the allocation strategy helpful?

            if it % 1024 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            info["time_spent"] = time.time() - start_time
            info["rand_prob"] = self.algo.get_random_action_prob(it)
            start_time = time.time()
            self.log(info, it, "train")
            if it % self.print_every == 0:
                logger.info(
                    f"iteration {it}: "
                    + " ".join(f"{k}:{v:.2f}" if (v <= 0 or v >= 1) else f"{k}:{v:.7f}" for k, v in info.items())
                )

            if valid_freq > 0 and it % valid_freq == 0:
                self.model.eval()
                if "seh" in self.cfg.log_dir:
                    threshold1 = 0.875
                    threshold2 = (0.875 + 1) / 2
                    threshold3 = 1
                elif "qm9" in self.cfg.log_dir:
                    threshold1 = 1.125
                    threshold2 = (1.125 + 1.25) / 2
                    threshold3 = 1.25
                else:
                    raise KeyError
                mols_at_least_thr1_info = read_all_results_at_least_reward(self.cfg.log_dir + "/train", threshold1)
                rdmols_at_least_thr1 = [
                    (row["r"], Chem.MolFromSmiles(row["smi"])) for _, row in mols_at_least_thr1_info.iterrows()
                ]
                self.log(
                    {
                        "modes_cnt_at_least_thr1": compute_num_modes(rdmols_at_least_thr1, threshold1),
                        "modes_cnt_at_least_thr2": compute_num_modes(rdmols_at_least_thr1, threshold2),
                        "modes_cnt_at_least_thr3": compute_num_modes(rdmols_at_least_thr1, threshold3),
                    },
                    it,
                    "train",
                )

                valid_db_path = self.cfg.log_dir + "/valid"
                if os.path.exists(valid_db_path):
                    for db_name in os.listdir(valid_db_path):
                        clear_all_data(f"{valid_db_path}/{db_name}")
                for batch in valid_dl:
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, "valid")
                    logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

                mols_info = read_all_results(self.cfg.log_dir + "/valid")
                val_rdmols = [Chem.MolFromSmiles(smile) for smile in mols_info["smi"]]
                val_rewards = [reward for reward in mols_info["r"]]
                val_rewards = np.array(val_rewards)
                assert (
                    val_rewards.shape[0] == self.cfg.num_validation_gen_steps * self.cfg.algo.valid_num_from_policy
                ), f"{val_rewards.shape[0]=}, {self.cfg.num_validation_gen_steps=}, {self.cfg.algo.valid_num_from_policy=}"

                self.log(
                    {
                        "top_k_diversity": top_k_diversity(val_rdmols, np.array(val_rewards), 100),
                        "top_k": get_topk(torch.Tensor(val_rewards), 100),
                        "diverse_top_k": compute_diverse_top_k(val_rdmols, torch.Tensor(val_rewards), 100),
                    },
                    it,
                    "valid",
                )

                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")

                # correlation is measured only on SEH task!
                test_gen_mols_n = 1000
                test_gen_graphs, test_gen_log_rewards = [], []
                while len(test_gen_graphs) < test_gen_mols_n:
                    trajs, _ = valid_dl.dataset.get_model_samples(
                        self.sampling_model, self.cfg.algo.valid_num_from_policy
                    )
                    test_gen_graphs.extend([traj["result"] for traj in trajs])
                    test_gen_log_rewards.extend([traj["log_reward"] for traj in trajs])
                test_gen_log_rewards = torch.Tensor(test_gen_log_rewards)
                try:
                    mean_corr, std_corr = monte_carlo_compute_correlation_stats(
                        self.algo,
                        self.ctx,
                        self.task,
                        self.sampling_model,
                        self.test_graphs,
                        self.test_mols_log_rewards,
                        self.device,
                        self.cfg.algo.do_parameterize_p_b,
                        self.cfg.algo.do_parameterize_p_b,
                        self.cfg.monte_carlo_corr_n,
                    )
                    self.log(
                        {
                            "corr": mean_corr,
                            "stdcorr": std_corr,
                        },
                        it,
                        "test",
                    )
                except Exception as exc:
                    print(exc)
                if "seh" in self.cfg.log_dir:
                    try:
                        mean_corr_gen, std_corr_gen = monte_carlo_compute_correlation_stats(
                            self.algo,
                            self.ctx,
                            self.task,
                            self.sampling_model,
                            test_gen_graphs,
                            test_gen_log_rewards,
                            self.device,
                            self.cfg.algo.do_parameterize_p_b,
                            self.cfg.algo.do_parameterize_p_b,
                            self.cfg.monte_carlo_corr_n,
                        )
                        self.log(
                            {
                                "corr_gen": mean_corr_gen,
                                "stdcorr_gen": std_corr_gen,
                            },
                            it,
                            "test",
                        )
                    except Exception as exc:
                        print(exc)
                if ckpt_freq > 0 and it % ckpt_freq == 0:
                    self._save_state(it)
        self._save_state(num_training_steps)

        num_final_gen_steps = self.cfg.num_final_gen_steps
        final_info = {}
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                range(num_training_steps + 1, num_training_steps + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                if hasattr(batch, "extra_info"):
                    for k, v in batch.extra_info.items():
                        if k not in final_info:
                            final_info[k] = []
                        if hasattr(v, "item"):
                            v = v.item()
                        final_info[k].append(v)
                if it % self.print_every == 0:
                    logger.info(f"Generating objs {it - num_training_steps}/{num_final_gen_steps}")
            final_info = {k: np.mean(v) for k, v in final_info.items()}

            logger.info("Final generation steps completed - " + " ".join(f"{k}:{v:.2f}" for k, v in final_info.items()))
            self.log(final_info, num_training_steps, "final")

        # for pypy and other GC having implementations, we need to manually clean up
        del train_dl
        del valid_dl
        if self.cfg.num_final_gen_steps:
            del final_dl

    def terminate(self):
        logger = logging.getLogger("logger")
        for handler in logger.handlers:
            handler.close()

        for hook in self.sampling_hooks:
            if hasattr(hook, "terminate") and hook.terminate not in self.to_terminate:
                hook.terminate()

        for terminate in self.to_terminate:
            terminate()

    def _save_state(self, it):
        state = {
            "models_state_dict": [self.model.state_dict()],
            "cfg": self.cfg,
            "step": it,
        }
        if self.sampling_model is not self.model:
            state["sampling_model_state_dict"] = [self.sampling_model.state_dict()]
        fn = pathlib.Path(self.cfg.log_dir) / "model_state.pt"
        with open(fn, "wb") as fd:
            torch.save(
                state,
                fd,
            )
        if self.cfg.store_all_checkpoints:
            shutil.copy(fn, pathlib.Path(self.cfg.log_dir) / f"model_state_{it}.pt")

    def log(self, info, index, key):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)
        if wandb.run is not None:
            wandb.log({f"{key}_{k}": v for k, v in info.items()}, step=index)

    def __del__(self):
        self.terminate()


def cycle(it):
    while True:
        for i in it:
            yield i
