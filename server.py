import time
import random
import logging as log

import ray
import torch
import numpy as np
from torch.utils.data import Dataset

import stats
import config
import client
import models
import data
import topology
import comparision_algorithms

import attacks_2 as attacks

import copy
import math

ModelState = tuple[models.StateDict, int]

class Server:
    def __init__(self, model: torch.nn.Module, client_types: list[str], client_splits: list[data.ClientSplit], num_clients: int, seed: int):
        self.model = model

        self.proj_dim = 20

        log.info("starting server...")
        D = sum(p.numel() for p in self.model.state_dict().values())

        log.debug("configuring generator")

        g = torch.Generator().manual_seed(seed)

        log.info("selecting dims")
        self.R = torch.randn(D, self.proj_dim, generator=g) / (self.proj_dim ** 0.5)

        self.clients = [
            client.Client(i, client_splits, self.project, model.state_dict(), client_types[i], seed=config.RANDOM_SEED) # pyright: ignore[reportArgumentType]
            #client.Client(i, client_splits, model.state_dict(), client_types[i], seed=config.RANDOM_SEED) # pyright: ignore[reportArgumentType]
            for i in range(num_clients)
        ]
        self.clients.sort(key=lambda c: c.cid)
        log.debug("Created Clients")

        self.global_cluster = topology.Cluster({c.cid: c for c in self.clients}, self.dist_func, self.project)

        log.info("finished initing server")
    
    def dist_func(self, client_model_state: torch.Tensor, global_model_state: torch.Tensor) -> float:
        return torch.norm(global_model_state - client_model_state, p=2).item()
    
    def project(self, vec: torch.Tensor) -> torch.Tensor:
        return (vec @ self.R)

    def train(self, dataset: Dataset, name_suffix: str='') -> str:
        
        name_suffix = '_' + name_suffix if name_suffix else ''
        run_id = time.asctime().replace(" ", "_").replace(":", "-")

        dataset_ref = ray.put(dataset)
        batch_size = ray.put(config.BATCH_SIZE)
        device = ray.put(config.DEVICE)

        for c in self.clients:
            if c.attack is not None:
                c.attack.prepare(self.model.state_dict(), models.get_model) # pyright: ignore[reportArgumentType]

        for round_id in range(1, config.ROUNDS+1):
            log.info(f"{round_id}/{config.ROUNDS}: ")
            # client_accs = self.round_fed_attract(round_id, run_id, dataset_ref, batch_size, device)
            # client_accs = self.round_fed_avg(round_id, run_id, dataset_ref, batch_size, device)
            # client_accs = self.round_fed_krum(round_id, run_id, dataset_ref, batch_size, device)
            client_accs = self.round_fed_cap(round_id, run_id, dataset_ref, batch_size, device)

            acc = stats.avg(client_accs)
            log.info(f"\tAccuracy: {acc*100:.2f}% | {len(client_accs)} total clients evaluated.")
            # log.info(f"\tMean Dist from Center: {self.global_cluster.ema_mean} | Std: {self.global_cluster.ema_var**0.5}")
            self.global_cluster.print_tree()

        return f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}"

    def round_fed_attract(self, round_id, run_id, dataset_ref: Dataset, batch_size, device) -> list[float]:
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        start = time.time()
        futures = [c.train(dataset_ref, batch_size, device) for c in selected]
        local_sds__cid = ray.get(futures)
        log.info(f"\t\tTime taken to train: {time.time() - start}")

        updated_states = {cid: sd[0] for sd, cid in local_sds__cid}

        start = time.time()
        for sd, cid in local_sds__cid:
            self.clients[cid].model_state = sd[0]
            self.clients[cid].model_state_flattened = self.project(stats.flatten(sd[0]))
        log.info(f"\t\tTime taken to copy updates: {time.time() - start}")

        if round_id % 5 == 0:
            start = time.time()
            self.global_cluster.split()
            log.info(f"\t\tTime taken to Split: {time.time() - start}")

        start = time.time()
        self.global_cluster.update_centers_upward(updated_states)
        log.info(f"\t\tTime taken to update : {time.time() - start}")

        start = time.time()
        self.global_cluster.propagate_downward()
        log.info(f"\t\tTime taken to propagate downwards: {time.time() - start}")

        log.info("Finished training. Starting Eval")

        accs = ray.get([c.evaluate(dataset_ref, batch_size, device) for c in self.clients if c.client_type == client.ClientTypes.NORMAL]) # list[tuple[float, client_id#int]]

        accs.sort(key=lambda x: x[1])
        accs_, _ = zip(*accs)

        accs_lis = list(accs_)

        return accs_lis
    
    def round_fed_avg(self, round_id, run_id, dataset_ref: Dataset, batch_size, device) -> list[float]:
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        start = time.time()
        futures = [c.train(dataset_ref, batch_size, device) for c in selected]
        local_sds__cid = ray.get(futures)
        log.info(f"\t\tTime taken to train: {time.time() - start}")

        updated_states = [(cid, sd[0]) for sd, cid in local_sds__cid]

        new_global = topology.Cluster.avg_model_states(updated_states)

        start = time.time()
        for c in self.clients:
            c.model_state = new_global
            # DONT NEED THIS: c.model_state_flattened = self.project(stats.flatten(sd[0]))
        log.info(f"\t\tTime taken to copy updates: {time.time() - start}")

        log.info("Finished training. Starting Eval")

        accs = ray.get([c.evaluate(dataset_ref, batch_size, device) for c in self.clients if c.client_type == client.ClientTypes.NORMAL]) # list[tuple[float, client_id#int]]
        
        # accs.sort(key=lambda x: x[1])
        accs_, _ = zip(*accs)

        accs_lis = list(accs_)

        return accs_lis
    
    def round_fed_krum(self, round_id, run_id, dataset_ref: Dataset, batch_size, device) -> list[float]:
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        start = time.time()
        futures = [c.train(dataset_ref, batch_size, device) for c in selected]
        local_sds__cid = ray.get(futures)
        log.info(f"\t\tTime taken to train: {time.time() - start}")

        updated_states = [(cid, sd[0]) for sd, cid in local_sds__cid]

        # new_global = topology.Cluster.avg_model_states(updated_states)
        new_global = comparision_algorithms.krum_aggregate_adaptive(updated_states)

        start = time.time()
        for c in self.clients:
            c.model_state = new_global
            # DONT NEED THIS: c.model_state_flattened = self.project(stats.flatten(sd[0]))
        log.info(f"\t\tTime taken to copy updates: {time.time() - start}")

        log.info("Finished training. Starting Eval")

        accs = ray.get([c.evaluate(dataset_ref, batch_size, device) for c in self.clients if c.client_type == client.ClientTypes.NORMAL]) # list[tuple[float, client_id#int]]
        
        # accs.sort(key=lambda x: x[1])
        accs_, _ = zip(*accs)

        accs_lis = list(accs_)

        return accs_lis
    
    def round_fed_cap(self, round_id, run_id, dataset_ref, batch_size, device) -> list[float]:
        """
        FedCAP: Robust Federated Learning via Customized Aggregation and Personalization
        Implements server-side customization, calibration, and anomaly detection.
        """
        
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
        selected_cids = {c.cid for c in selected}
        
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
                        sim = self._cosine_similarity_flat(d_k_flat, other_d_flat)
                        similarities[other_cid] = sim
                    
                    # Compute aggregation weights with softmax normalization (Eq. 5)
                    weights = self._compute_customized_weights(
                        cid, similarities, self.fedcap_alpha, self.fedcap_phi
                    )
                    
                    # Aggregate recovered models from previous round using weights (Eq. 3)
                    aggregated = self._aggregate_weighted_states(
                        self.recovered_model_pool, weights
                    )
                    customized_models[cid] = aggregated if aggregated is not None else copy.deepcopy(self.global_model_state)
        
        # =========================================================================
        # STEP 3: Distribute and Local Training
        # =========================================================================
        for c in selected:
            # Load customized model into client
            c.model_state = copy.deepcopy(customized_models[c.cid])
        
        futures = [c.train(dataset_ref, batch_size, device) for c in selected]
        local_results_pre = ray.get(futures)  # List of (state_dict, cid)
        local_results = [(i[0][0], i[1]) for i in local_results_pre]
        
        # =========================================================================
        # STEP 4: Recovery and Calibration (Section V-C)
        # Recovery: \tilde{w}_k^t = \hat{w}_k^t + d_k^t (returned state)
        # Calibration: \tilde{d}_k^t = \tilde{w}_k^t - w^t (aligned to global reference)
        # =========================================================================
        recovered_models = {}      # \tilde{w}_k^t
        calibrated_updates = {}    # Flattened \tilde{d}_k^t
        
        for state_dict, cid in local_results:
            # Recovery: the returned state is the recovered model
            recovered_models[cid] = state_dict
            
            # Calibration: difference between recovered model and current global model
            diff_state = self._state_diff(state_dict, self.global_model_state)
            calibrated_updates[cid] = stats.flatten(diff_state) # pyright: ignore[reportArgumentType]
        
        # =========================================================================
        # STEP 5: Anomaly Detection (Section V-C)
        # Detect malicious clients by thresholding Euclidean norm of calibrated updates
        # =========================================================================
        newly_detected = []
        for cid, cal_flat in calibrated_updates.items():
            if cid in self.detected_malicious:
                continue
            norm = torch.norm(cal_flat).item()
            if norm > self.fedcap_T_norm:
                self.detected_malicious.add(cid)
                newly_detected.append((cid, norm))
        
        if newly_detected:
            log.info(f"Round {round_id}: Detected and removed malicious clients: {newly_detected}")
        
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
        benign_clients = [
            c for c in self.clients 
            if c.client_type == client.ClientTypes.NORMAL
        ]
        
        if not benign_clients:
            log.warning(f"Round {round_id}: No benign clients remaining for evaluation")
            return []
        
        eval_futures = [c.evaluate(dataset_ref, batch_size, device) for c in benign_clients]
        eval_results = ray.get(eval_futures)
        # eval_results.sort(key=lambda x: x[1])
        accs, _ = zip(*eval_results)
        return list(accs)


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