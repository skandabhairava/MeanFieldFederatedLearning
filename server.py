from collections import Counter
import time
import random
import logging as log
from functools import reduce
from enum import Enum
import sys

import ray
import torch
from torch.utils.data import Dataset
import numpy as np

import stats
import config
import client
import models
import data
import topology
import comparision_algorithms
from client_types import ClientTypes
from attacks import ALIEAttack, IPMAttack, LabelSwitchAttack

import copy
import math

ModelState = tuple[models.StateDict, int]

def find_size(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)

    # Avoid double-counting the same object
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(
            find_size(k, seen) + find_size(v, seen)
            for k, v in obj.items()
        )

    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(find_size(item, seen) for item in obj)

    elif hasattr(obj, "__dict__"):
        size += find_size(obj.__dict__, seen)

    elif hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            try:
                size += find_size(getattr(obj, slot), seen)
            except AttributeError:
                pass

    return size

class TrainProtocol(Enum):
    FedAttract = 1
    FedAvg = 2
    FedKrum = 3
    FedCap = 4

class Server:
    def __init__(
            self, 
            model: torch.nn.Module, 
            client_types: list[ClientTypes], 
            client_splits: list[data.ClientSplit], 
            num_clients: int,
            spill_to_disk: bool,
            spill_folder: str,
            seed: int,
        ):
        self.model = model
        self.spill_to_disk = spill_to_disk
        self.spill_folder = spill_folder

        if self.spill_to_disk:
            print(f"Spilling Client states to {spill_folder} when required.")

        if config.USE_VELOCITY:
            self.total_proj_dim = 20
            self.proj_dim = 5
            self.zero_proj = torch.zeros((5,))
        else:
            # self.total_proj_dim = 20
            self.proj_dim = 20
            # self.zero_proj = torch.zeros((5,))


        log.info("starting server...")
        D = sum(p.numel() for p in self.model.state_dict().values())

        log.debug("configuring generator")

        g = torch.Generator().manual_seed(seed)

        log.info("selecting dims")
        self.R = torch.randn(D, self.proj_dim, generator=g) / (self.proj_dim ** 0.5)

        self.clients = [
            client.Client(
                i, 
                client_splits, 
                self.project, 
                model.state_dict(), # pyright: ignore[reportArgumentType]
                self.spill_folder,
                client_types[i],
                seed=config.RANDOM_SEED
            )
            for i in range(num_clients)
        ]
        self.clients.sort(key=lambda c: c.cid)
        log.debug("Created Clients")

        self.client_types = Counter([c.client_type for c in self.clients])

        self.global_cluster = topology.Cluster(
            {c.cid: c for c in self.clients}, 
            self.dist_func, 
            self.project,
            self.spill_folder
        )

        self.W = models.get_size() # |w| = number of model params
        self.W_flat = len(self.global_cluster.get_model_state_flattened())

        self.ALIE_clients = [c for c in self.clients if c.client_type == ClientTypes.ALIE or c.client_type == ClientTypes.LATE_ALIE]
        self.IPM_clients = [c for c in self.clients if c.client_type == ClientTypes.IPM or c.client_type == ClientTypes.LATE_IPM]
        self.SUBTLE_clients = [c for c in self.clients if c.client_type == ClientTypes.SUBTLE or c.client_type == ClientTypes.LATE_SUBTLE]
        self.HONEST_clients = [c for c in self.clients if c.client_type == ClientTypes.NORMAL]

        log.info("finished initing server")
    
    def dist_func(self, client_model_state: torch.Tensor, global_model_state: torch.Tensor, only_l2: bool, max_l2_dist: float|None=None) -> float:
        # if not config.USE_VELOCITY or only_l2:
        #     return torch.norm(global_model_state[:self.proj_dim] - client_model_state[:self.proj_dim], p=2).item()
        assert torch.isfinite(global_model_state).all(), "global state has NaN/Inf"
        assert torch.isfinite(client_model_state).all(), "client state has NaN/Inf"

        return torch.norm(global_model_state - client_model_state, p=2).item()
    
    def project(self, vec: torch.Tensor, old_proj: torch.Tensor|None=None, shift_diffs: bool=False, add_to_current_diff: bool=False, disp: bool=False) -> torch.Tensor:
        assert torch.isfinite(vec).all(), "projection: state has NaN/Inf"
        if disp:
            return (vec @ self.DISP_R)

        if not config.USE_VELOCITY:
            return (vec @ self.R)

        if old_proj is None:
            new = (vec @ self.R)
            return torch.cat([new, self.zero_proj, self.zero_proj, self.zero_proj, torch.norm(new).unsqueeze(dim=-1)])

        if not shift_diffs:
            new = (vec @ self.R)
            if add_to_current_diff:
                diff = new - old_proj[:self.proj_dim]
                old_proj[(self.total_proj_dim - self.proj_dim) : self.total_proj_dim] += diff

            old_proj[:self.proj_dim] = new
            old_proj[-1] = torch.norm(new)
            return old_proj

        new = (vec @ self.R)
        diff = new - old_proj[:self.proj_dim]
        old_proj[:self.proj_dim] = new

        for i in range(1, self.total_proj_dim//self.proj_dim - 1):
            old_proj[(i*self.proj_dim) : (i*self.proj_dim + self.proj_dim)] = old_proj[(i*self.proj_dim + self.proj_dim): (i*self.proj_dim + 2*self.proj_dim)]

        old_proj[(self.total_proj_dim - self.proj_dim) : self.total_proj_dim] = diff
        old_proj[-1] = torch.norm(new)

        return old_proj

    @staticmethod
    @torch.no_grad()
    def get_norm(state: models.StateDict, original_state: models.StateDict):
        
        keys = [
            k for k in state
            if torch.is_floating_point(state[k])
        ]

        dist_sq = torch.stack([
            (state[k] - original_state[k]).pow(2).sum()
            for k in keys
        ]).sum()

        dist = torch.sqrt(dist_sq)

        return dist.item()

    @staticmethod
    @torch.no_grad()
    def clip_state_dict_distance(state: models.StateDict, original_state: models.StateDict, max_norm: float):
        """
        Clip ||state - original_state||_2 to max_norm.

        Modifies `state` in-place and returns the original norm.
        """

        keys = [
            k for k in state
            if torch.is_floating_point(state[k])
        ]

        dist_sq = torch.stack([
            (state[k] - original_state[k]).pow(2).sum()
            for k in keys
        ]).sum()

        dist = torch.sqrt(dist_sq)

        if dist > max_norm:
            scale = max_norm / dist

            for k in keys:
                state[k].copy_(
                    original_state[k] +
                    scale * (state[k] - original_state[k])
                )

        return dist.item()

    def switch_clients_to_disk(self):
        for c in self.clients:
            c.save_model_state()
            c.save_model_state_flattened()
            c.in_memory_swapped = False
            c.model_state = None
            c.model_state_flattened = None

    def switch_clients_to_memory(self):
        for c in self.clients:
            c.model_state = c.get_model_state()
            c.model_state_flattened = c.get_model_state_flattened()
            c.in_memory_swapped = True

    def run_generalization_matrix_eval(self, dataset_ref: Dataset, batch_size: int, device: str, log_save_dir: str):
        eval_runs_out = []
        for ii, c1 in enumerate(self.clients, start=1):
            eval_runs = []
            for c2 in self.clients:
                eval_runs.append(c1.evaluate(dataset_ref, batch_size, device, should_attack=False, calc_counts=False, split=c2.split, extra=c2.cid))
            
            eval_runs_out.extend(ray.get(eval_runs))
            log.info(f"{ii*len(self.clients)}/{len(self.clients)**2} Done.")
        # accs, cids, classes, out_cid = zip(*eval_runs_out)

        out_dict = {}
        for (acc, main_cid, _clss, sec_cid) in eval_runs_out:
            res = out_dict.setdefault(main_cid, {}).setdefault(sec_cid, {})
            res["acc"] = acc*100
            res["main_client_type"] = str(self.clients[main_cid].client_type)
            res["sec_client_type"] = str(self.clients[sec_cid].client_type)
            log.info(f"{main_cid}|{res["main_client_type"]} client w/ {sec_cid}|{res["sec_client_type"]}: {acc*100:.2f}%")

        with open(f"{log_save_dir}/generalization.json", "w") as f:
            import json
            json.dump(out_dict, f)

    def train(self, train_protocol: TrainProtocol, dataset: Dataset, run_id: str, log_save_dir: str, save: bool = False, test_run_calc:bool=False) -> str:
        if self.spill_to_disk and train_protocol == TrainProtocol.FedAttract:
            self.global_cluster.swap_to_disk_checkpoint_recursive()

        dataset_ref = dataset
        batch_size = config.BATCH_SIZE
        device = config.DEVICE
        if not test_run_calc:
            dataset_ref = ray.put(dataset)
            batch_size = ray.put(config.BATCH_SIZE)
            device = ray.put(config.DEVICE)

            log.info("Preparing attacks", extra={"save": True})

            for c in self.clients:
                if c.attack is not None:
                    c.attack.prepare(self.model.state_dict(), models.get_model, dataset, c.split) # pyright: ignore[reportArgumentType]
        
        bigo_ts = []
        bigo_ss = []

        for round_id in range(1, config.ROUNDS+1):
            if len(self.ALIE_clients) != 0:
                ALIEAttack.global_model_state = topology.Cluster.avg_model_states([(c.cid, c.get_model_state()) for c in self.clients])  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            if len(self.IPM_clients) != 0:
                IPMAttack.global_model_state = topology.Cluster.avg_model_states(
                    [(c.cid, c.get_model_state()) for c in self.clients if c.attack is None]
                )

            log.info(f"{round_id}/{config.ROUNDS}: ", extra={"save": True})
            if train_protocol == TrainProtocol.FedAttract:
                time_taken, client_accs, bigo_t, bigo_s = self.round_fed_attract(
                    round_id, 
                    run_id, 
                    dataset_ref, 
                    batch_size, 
                    self.ALIE_clients[:1] + self.IPM_clients[:1],
                    device, 
                    test_run_calc
                )
            elif train_protocol == TrainProtocol.FedAvg:
                time_taken, client_accs, bigo_t, bigo_s = self.round_fed_avg(
                    round_id, 
                    run_id, 
                    dataset_ref, 
                    batch_size, 
                    self.ALIE_clients[:1] + self.IPM_clients[:1],
                    device
                )
            elif train_protocol == TrainProtocol.FedKrum:
                time_taken, client_accs, bigo_t, bigo_s = self.round_fed_krum(
                    round_id, 
                    run_id, 
                    dataset_ref, 
                    batch_size, 
                    self.ALIE_clients[:1] + self.IPM_clients[:1],
                    device,
                )
            elif train_protocol == TrainProtocol.FedCap:
                time_taken, client_accs, bigo_t, bigo_s = self.round_fed_cap(
                    round_id, 
                    run_id, 
                    dataset_ref, 
                    batch_size, 
                    self.ALIE_clients[:1] + self.IPM_clients[:1],
                    device, 
                    test_run_calc
                )

            if not test_run_calc:
                log.info(f"Time taken to train: {time_taken[0]}s, accuracy: {time_taken[2]}", extra={"save": True})

            log.info(f"Big O time: {bigo_t} | Big O space: {bigo_s} | Time taken algorithm model: {time_taken[1]}", extra={"save": True})
            bigo_ts.append(bigo_t)
            bigo_ss.append(bigo_s)

            if not test_run_calc:
                acc = stats.avg(client_accs)
                log.info(f"\tAccuracy: {acc*100:.2f}% | {len(client_accs)} total clients evaluated.", extra={"save": True})

            self.global_cluster.print_tree()

            # if len(ALIE_clients) != 0:
            #     ALIEAttack.perform_general_post_update_attack(self.clients)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

            if not test_run_calc:
                if round_id % 10 == 0 and round_id != config.ROUNDS:
                    time.sleep(60*10)
 
        avg_big_ot = stats.avg(bigo_ts) # pyright: ignore[reportPossiblyUnboundVariable]
        avg_big_os = stats.avg(bigo_ss) # pyright: ignore[reportPossiblyUnboundVariable]
        log.info(f"\tAvg BigO time: {avg_big_ot} | Avg BigO space: {avg_big_os}", extra={"save": True})
        log.info(f"\tMax BigO time: {max(bigo_ts)} | Max BigO space: {max(bigo_ss)}", extra={"save": True}) # pyright: ignore[reportPossiblyUnboundVariable]
        log.info(f"\tTotal BigO time: {sum(bigo_ts)}", extra={"save": True}) # pyright: ignore[reportPossiblyUnboundVariable]

        if save and not test_run_calc:
            for c in self.clients:
                c.save(log_save_dir)

            if train_protocol == TrainProtocol.FedAttract:
                self.global_cluster.save_metadata(log_save_dir, "topology")

            if ClientTypes.LABEL_SWITCH in self.client_types.keys():
                with open(f"{log_save_dir}/label_switch.json", "w") as f:
                    import json
                    json.dump(LabelSwitchAttack.label_switch, f)

        if self.spill_to_disk:
            self.global_cluster.clean_checkpoints()

        return log_save_dir

    def round_fed_attract(
            self, 
            round_id: int, 
            run_id: str, 
            dataset_ref: Dataset, 
            batch_size: int, 
            post_update_attack_clients: list[client.Client],
            device: str, 
            test_run_calc:bool=False
        ) -> tuple[tuple[float, float, float], list[float], int, int]:
        bigo_t = 0
        bigo_s = 0
        if not test_run_calc:
            m = int(len(self.clients) * config.CLIENT_FRAC)
            selected = random.sample(self.clients, m)

            if self.spill_to_disk:
                self.switch_clients_to_memory()
            log.info(f"Clients in memory AFTER switching to memory costs {find_size(self.clients)} MB")
            log.info(f"Switching to memory worked? {all(c.model_state is not None for c in self.clients)}")

            train_start = time.time()
            futures = [c.train(dataset_ref, batch_size, device, round_id) for c in selected]
            local_sds__cid = ray.get(futures)

            train_end = time.time()
            old_models = {}

            for sd, cid in local_sds__cid:
                self.clients[cid].model_state = sd[0]
                self.clients[cid].model_state_flattened = self.project(
                    stats.flatten(sd[0]),
                    old_proj=self.clients[cid].model_state_flattened,
                    shift_diffs=True
                )

            print(f"attacks passed to fedattract: {post_update_attack_clients}, round: {round_id}")
            for attack_client in post_update_attack_clients:
                if attack_client.attack is not None:
                    attack_client.attack.perform_general_post_update_attack(
                        self.clients,
                        len(self.clients),
                        self.client_types[attack_client.atack_type]
                    )

            

        alg_start = time.time()
        if round_id % config.ATTRACT_SPLIT_EVERY == 0:
            bigo_t += self.global_cluster.split(self.W, self.W_flat)

        bigo_t += self.global_cluster.update_centers_upward(self.W, self.W_flat)
        bigo_s_, bigo_t_ = self.global_cluster.propagate_downward(recalc_dists=False, bigo_W_size=self.W, bigo_W_flat_size=self.W_flat)
        bigo_s += bigo_s_
        bigo_t += bigo_t_
        alg_end = time.time()

        time.sleep(10)
        log.info(f"Clients in memory AFTER switching to disk costs {find_size(self.clients)} MB")
        log.info(f"Switching to disk worked? {all(c.model_state is None for c in self.clients)}")

        if not test_run_calc:
            log.info("Finished training. Starting Eval")
            if self.spill_to_disk:
                self.switch_clients_to_memory()
            acc_start = time.time()
            accs = ray.get([c.evaluate(dataset_ref, batch_size, device) for c in self.clients if c.client_type == client.ClientTypes.NORMAL]) # list[tuple[float, client_id#int]]
            acc_end = time.time()
            if self.spill_to_disk:
                self.switch_clients_to_disk()

            accs.sort(key=lambda x: x[1])
            accs_, _, counters, _extra = zip(*accs)

            if counters[0] is not None:
                final_counter = reduce(lambda x, y: x+y, counters)
                log.info(final_counter)

            accs_lis = list(accs_)

        if test_run_calc:
            return (0, alg_end-alg_start, 0), [], bigo_t, bigo_s
        return (train_end-train_start, alg_end-alg_start, acc_end-acc_start), accs_lis, bigo_t, bigo_s # pyright: ignore[reportPossiblyUnboundVariable, reportOperatorIssue]

    def round_fed_avg(
            self, 
            round_id, 
            run_id, 
            dataset_ref: Dataset, 
            batch_size, 
            post_update_attack_clients: list[client.Client],
            device
        ) -> tuple[tuple[float, float, float], list[float], int, int]:
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        train_start = time.time()
        futures = [c.train(dataset_ref, batch_size, device, round_id) for c in selected]
        local_sds__cid = ray.get(futures)
        train_end = time.time()

        for (state, _), cid in local_sds__cid:
            self.clients[cid].model_state = state

        for attack_client in post_update_attack_clients:
            if attack_client.attack is not None:
                attack_client.attack.perform_general_post_update_attack(
                    self.clients,
                    len(self.clients),
                    self.client_types[attack_client.client_type],
                    round_id
                )

        alg_start = time.time()
        new_global = topology.Cluster.avg_model_states([(c.cid, c.get_model_state()) for c in self.clients])
        alg_end = time.time()

        # start = time.time()
        for c in self.clients:
            c.model_state = new_global
            # DONT NEED THIS: c.model_state_flattened = self.project(stats.flatten(sd[0]))
        # log.info(f"\t\tTime taken to copy updates: {time.time() - start}")

        log.info("Finished training. Starting Eval")

        acc_start = time.time()
        accs = ray.get([c.evaluate(dataset_ref, batch_size, device) for c in self.clients if c.client_type == client.ClientTypes.NORMAL]) # list[tuple[float, client_id#int]]
        acc_end = time.time()
        
        accs_, _, _, _ = zip(*accs)

        accs_lis = list(accs_)

        big_o_t = m
        big_o_s = len(self.clients)*models.get_size()

        return (train_end-train_start, alg_end-alg_start, acc_end-acc_start), accs_lis, big_o_t, big_o_s

    # time_taken, client_accs, bigo_t, bigo_s    
    def round_fed_krum(
            self, 
            round_id, 
            run_id, 
            dataset_ref: Dataset, 
            batch_size,
            post_update_attack_clients: list[client.Client],
            device
        ) -> tuple[tuple[float, float, float], list[float], int, int]:
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        train_start = time.time()
        futures = [c.train(dataset_ref, batch_size, device, round_id) for c in selected]
        local_sds__cid = ray.get(futures)
        train_end = time.time()

        for (state, _), cid in local_sds__cid:
            self.clients[cid].model_state = state

        for attack_client in post_update_attack_clients:
            if attack_client.attack is not None:
                attack_client.attack.perform_general_post_update_attack(
                    self.clients,
                    len(self.clients),
                    self.client_types[attack_client.client_type],
                    round_id
                )

        alg_start = time.time()
        new_global = comparision_algorithms.krum_aggregate_adaptive([(c.cid, c.model_state) for c in self.clients])
        alg_end = time.time()

        log.info(f"Client Selected Krum: {client_sel} | {self.clients[client_sel].client_type if client_sel >= 0 else 'median selected'}")

        # start = time.time()
        for c in self.clients:
            c.model_state = new_global
            # DONT NEED THIS: c.model_state_flattened = self.project(stats.flatten(sd[0]))
        # log.info(f"\t\tTime taken to copy updates: {time.time() - start}")

        log.info("Finished training. Starting Eval")

        acc_start = time.time()
        accs = ray.get([c.evaluate(dataset_ref, batch_size, device) for c in self.clients if c.client_type == client.ClientTypes.NORMAL]) # list[tuple[float, client_id#int]]
        acc_end = time.time()
        
        # accs.sort(key=lambda x: x[1])
        accs_, _, _, _ = zip(*accs)

        accs_lis = list(accs_)

        return (train_end-train_start, alg_end-alg_start, acc_end-acc_start), accs_lis, m, 1
    
    def round_fed_cap(
            self, 
            round_id, 
            run_id, 
            dataset_ref, 
            batch_size, 
            post_update_attack_clients: list[client.Client],
            device, 
            test_run_calc:bool=False
        ) -> tuple[tuple[float, float, float], list[float], int, int]:
        """
        FedCAP: Robust Federated Learning via Customized Aggregation and Personalization
        Implements server-side customization, calibration, and anomaly detection.
        """
        bigo_t = 0
        bigo_s  = 0
        # Initialize FedCAP state containers on first round
        if round_id == 1:
            self.recovered_model_pool: dict[int, models.StateDict] = {}      # Stores \tilde{w}^{t-1} from previous round
            self.calibrated_update_pool: dict[int, torch.Tensor] = {}    # Stores flattened \tilde{d}^{t-1} 
            self.global_model_state: models.StateDict = copy.deepcopy(self.model.state_dict()) # pyright: ignore[reportAttributeAccessIssue]
            self.detected_malicious = set()     # Permanently detected malicious clients
            
            # FedCAP hyperparameters (can be moved to config)
            self.fedcap_alpha = getattr(config, 'FEDCAP_ALPHA', 5.0)      # Scale factor for softmax
            self.fedcap_phi = getattr(config, 'FEDCAP_PHI', 0.2)          # Self-contribution weight
            self.fedcap_T_norm = getattr(config, 'FEDCAP_T_NORM', 10.0)   # Detection threshold
        
        # Client selection
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        alg_start = time.time()
        
        # =========================================================================
        # STEP 1: Global Model Update (using previous round's recovered models)
        # =========================================================================
        if round_id > 1 and self.recovered_model_pool:
            # Aggregate benign recovered models from round t-1 to get w^t
            benign_pool: dict[int, models.StateDict] = {
                cid: model for cid, model in self.recovered_model_pool.items() 
                if cid not in self.detected_malicious
            }
            if benign_pool:
                self.global_model_state = topology.Cluster.avg_model_states(list(benign_pool.items()))
        
        # =========================================================================
        # STEP 2: Model Customization (Section V-A)
        # Assign each client k a customized model \hat{w}_k^t based on similarity
        # with historical calibrated updates
        # =========================================================================
        customized_models = {}
        
        if round_id == 1 or not self.recovered_model_pool:
            # First round: all clients start from global model
            for c in selected:
                customized_models[c.cid] = copy.deepcopy(self.global_model_state)
        else:
            for c in selected:
                cid = c.cid
                if cid not in self.calibrated_update_pool:
                    # New client: no historical update available, use global model
                    customized_models[cid] = copy.deepcopy(self.global_model_state)
                else:
                    # Compute similarities between this client's previous update and all others
                    d_k_flat = self.calibrated_update_pool[cid]
                    similarities = {}
                    
                    for other_cid, other_d_flat in self.calibrated_update_pool.items():
                        if other_cid == cid or other_cid in self.detected_malicious:
                            continue

                        bigo_t += self.W

                        sim = self._cosine_similarity_flat(d_k_flat, other_d_flat)
                        similarities[other_cid] = sim

                    # Compute aggregation weights with softmax normalization (Eq. 5)
                    bigo_t += 2*len(similarities)
                    weights = self._compute_customized_weights(
                        cid, similarities, self.fedcap_alpha, self.fedcap_phi
                    )

                    bigo_t += len(self.recovered_model_pool)*self.W
                    # Aggregate recovered models from previous round using weights (Eq. 3)
                    aggregated = self._aggregate_weighted_states(
                        self.recovered_model_pool, weights
                    )
                    customized_models[cid] = aggregated if aggregated is not None else copy.deepcopy(self.global_model_state)
        
        # =========================================================================
        # STEP 3: Distribute and Local Training
        # =========================================================================
        alg_pause = time.time()

        for c in selected:
            # Load customized model into client
            c.model_state = copy.deepcopy(customized_models[c.cid])

        if not test_run_calc:
            train_start = time.time()
            futures = [c.train(dataset_ref, batch_size, device, round_id) for c in selected]
            local_results_pre = ray.get(futures)  # List of (state_dict, cid)
            train_end = time.time()

            for (state, _), cid in local_results_pre:
                self.clients[cid].model_state = state

            for attack_client in post_update_attack_clients:
                if attack_client.attack is not None:
                    attack_client.attack.perform_general_post_update_attack(
                        self.clients,
                        len(self.clients),
                        self.client_types[attack_client.client_type],
                        round_id
                    )

            local_results = [(c.get_model_state(), c.cid) for c in self.clients]

        alg_continue = time.time()
        
        # =========================================================================
        # STEP 4: Recovery and Calibration (Section V-C)
        # Recovery: \tilde{w}_k^t = \hat{w}_k^t + d_k^t (returned state)
        # Calibration: \tilde{d}_k^t = \tilde{w}_k^t - w^t (aligned to global reference)
        # =========================================================================
        recovered_models = {}      # \tilde{w}_k^t
        calibrated_updates: dict[int, torch.Tensor] = {}    # Flattened \tilde{d}_k^t

        # bigo_t += len(local_results) # pyright: ignore[reportPossiblyUnboundVariable]
        # # calibrate updates
        # if not test_run_calc:
        #     for state_dict, cid in local_results: # pyright: ignore[reportPossiblyUnboundVariable]
        #         # Recovery: the returned state is the recovered model
        #         recovered_models[cid] = state_dict
                
        #         # Calibration: difference between recovered model and current global model
        #         diff_state = self._state_diff(state_dict, self.global_model_state)
        #         calibrated_updates[cid] = stats.flatten(diff_state) # pyright: ignore[reportArgumentType]
        # else:
        #     for client_ in self.clients:
        #         recovered_models[client_.cid] = client_.model_state
        #         diff_state = self._state_diff(client_.get_model_state(), self.global_model_state)
        #         calibrated_updates[client_.cid] = stats.flatten(diff_state) # pyright: ignore[reportArgumentType]
        # =========================================================================
        # STEP 5: Anomaly Detection (Section V-C)
        # Detect malicious clients by thresholding Euclidean norm of calibrated updates
        # =========================================================================
        avg = stats.flatten(self.global_model_state)
        direction = avg / avg.norm()
        avg_coord = torch.dot(avg, direction).item()

        log.info(f"\tGLOBAL: {avg_coord}")
        newly_detected = []

        if not test_run_calc:
            for state_dict, cid in local_results: # pyright: ignore[reportPossiblyUnboundVariable]
                # Recovery: the returned state is the recovered model
                recovered_models[cid] = state_dict
                
                # Calibration: difference between recovered model and current global model
                bigo_t += 1
                diff_state = self._state_diff(state_dict, self.global_model_state)
                calibrated_updates[cid] = stats.flatten(diff_state) # pyright: ignore[reportArgumentType]

                if cid in self.detected_malicious:
                    continue

                bigo_t += calibrated_updates[cid].numel()
                norm = torch.norm(calibrated_updates[cid]).item()
                if norm > self.fedcap_T_norm:
                    self.detected_malicious.add(cid)
                    newly_detected.append((cid, norm))

                upd = calibrated_updates[cid] + avg
                log.info(f"{cid=} {torch.dot(upd.flatten(), direction).item()} {norm=}")
        else:
            for client_ in self.clients:
                recovered_models[client_.cid] = client_.model_state
                diff_state = self._state_diff(client_.get_model_state(), self.global_model_state)
                calibrated_updates[client_.cid] = stats.flatten(diff_state) # pyright: ignore[reportArgumentType]
        
        if newly_detected:
            log.info(f"Round {round_id}: Detected and removed malicious clients: {newly_detected}  {len(newly_detected)=}", extra={"save": True})
        
        # =========================================================================
        # STEP 6: Update Historical Pools for next round
        # Only store benign clients; malicious are permanently removed
        # =========================================================================
        self.recovered_model_pool = {
            cid: recovered_models[cid] 
            for cid in recovered_models 
            if cid not in self.detected_malicious
        }
        self.calibrated_update_pool = {
            cid: calibrated_updates[cid] 
            for cid in calibrated_updates 
            if cid not in self.detected_malicious
        }
        
        # =========================================================================
        # STEP 7: Evaluation (only benign clients)
        # =========================================================================
        
        alg_end = time.time()

        if not test_run_calc:
            acc_start = time.time()
            eval_futures = [
                c.evaluate(dataset_ref, batch_size, device) 
                for c in self.clients
                if c.client_type == client.ClientTypes.NORMAL
            ]
            eval_results = ray.get(eval_futures)
            acc_end = time.time()
            # eval_results.sort(key=lambda x: x[1])
            accs, _, _ = zip(*eval_results)

        bigo_s = (len(self.calibrated_update_pool) + len(self.recovered_model_pool) + 1) * self.W

        if test_run_calc:
            return (0, (alg_end-alg_continue)+(alg_pause-alg_start), 0), [], bigo_t, bigo_s # pyright: ignore[reportPossiblyUnboundVariable]
        return (train_end-train_start, (alg_end-alg_continue)+(alg_pause-alg_start), acc_end-acc_start), list(accs), bigo_t, bigo_s # pyright: ignore[reportPossiblyUnboundVariable, reportOperatorIssue]


    # =============================================================================
    # Helper Methods (add these to the Server class)
    # =============================================================================

    def _cosine_similarity_flat(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute cosine similarity between two flattened vectors."""
        dot = torch.dot(vec1, vec2).item()
        norm1 = torch.norm(vec1).item()
        norm2 = torch.norm(vec2).item()
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


    def _compute_customized_weights(self, target_cid: int, similarities: dict, alpha: float, phi: float) -> dict[int, float]:
        """
        Compute aggregation weights p'_{k,i} using softmax normalization (Eq. 4-5).
        - target_cid gets weight phi (self-contribution)
        - Others share (1-phi) weighted by exp(alpha * similarity)
        """

        # alpha = 5
        # phi = 0.2

        if not similarities:
            return {target_cid: 1.0}
        
        # Softmax over similarities
        exp_sims = {cid: math.exp(alpha * sim) for cid, sim in similarities.items()}
        sum_exp = sum(exp_sims.values())
        
        weights: dict[int, float] = {}
        for cid, exp_sim in exp_sims.items():
            weights[cid] = (1 - phi) * exp_sim / sum_exp
        
        # Self-contribution
        weights[target_cid] = phi
        return weights


    def _aggregate_weighted_states(self, state_dicts: dict[int, models.StateDict], weights: dict[int, float]) -> models.StateDict|None:
        """Weighted aggregation of model state dictionaries."""
        if not weights or not state_dicts:
            return None
        
        result = None
        total_weight = 0.0
        
        for cid, state in state_dicts.items():
            if cid not in weights:
                continue
            w = weights[cid]
            if result is None:
                result = {k: v.clone() * w for k, v in state.items()}
            else:
                for k in result:
                    result[k] += state[k] * w
            total_weight += w
        
        if result and total_weight > 0:
            for k in result:
                result[k] /= total_weight

        return result # pyright: ignore[reportReturnType]


    def _state_diff(self, state1: models.StateDict, state2: models.StateDict) -> dict:
        """Element-wise difference between two state dicts (state1 - state2)."""
        return {k: state1[k] - state2[k] for k in state1}