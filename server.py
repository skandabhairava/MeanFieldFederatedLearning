import gc
import os
from typing import Sequence
import time
import pickle
import random
import logging as log
from dataclasses import dataclass, asdict

import ray
import torch
import numpy as np
from torch.utils.data import Dataset

import stats
import config
import client
import models
import attacks

from typing import Callable

ModelState = tuple[models.StateDict, int]

@dataclass
class RoundResults:
    round_id: int
    selected_clients: list[int]
    client_accs: list[float]
    delta_proj: list[np.ndarray]
    global_post_train_proj: np.ndarray
    debug_client_types: list[str]

class Server:
    def __init__(self, model: torch.nn.Module, clients: list[client.Client], seed: int):
        self.model = model
        self.clients = clients
        self.clients.sort(key=lambda c: c.cid)

        self.proj_dim = 200

        log.info("starting server...")
        D = sum(p.numel() for p in self.model.state_dict().values())

        log.debug("configuring generator")

        g = torch.Generator().manual_seed(seed)

        log.info("selecting dims")
        self.R = torch.randn(D, self.proj_dim, generator=g) / (self.proj_dim ** 0.5)

        log.info("finished initing server")
    
    # FedAVG
    def fed_avg_aggregate(self, client_weights: Sequence[ModelState]) -> ModelState:
        # total all of weights
        total = 0
        for i in range(len(client_weights)):
            total += client_weights[i][1]
        
        global_weights = client_weights[0][0]
        for key in global_weights.keys(): # for each param in clients
            for i in range(len(client_weights)): # for each client
                if i == 0: # scale global_wights(client i == 0)'s by itself
                    global_weights[key] = (client_weights[0][1] / total) * global_weights[key]
                else: # add a scaled value to the avg/global weight client
                    w = client_weights[i][1] / total
                    global_weights[key] = global_weights[key] + w * client_weights[i][0][key]
        return global_weights, total
    
    # FedAttractAVG
    def fed_attract_aggregate(self, client_weights: Sequence[ModelState], client_dists: list[tuple[float, int]], total_client_dist: float) -> ModelState:
        # total all of weights
        total = 0
        for i in range(len(client_weights)):
            total += client_weights[i][1]
        
        global_weights = client_weights[0][0]
        for key in global_weights.keys(): # for each param in clients
            for i in range(len(client_weights)): # for each client
                if i == 0: # scale global_wights(client i == 0)'s by itself
                    global_weights[key] = (client_weights[0][1] / total) * (1-(client_dists[0][0] / total_client_dist)) * global_weights[key]
                else: # add a scaled value to the avg/global weight client
                    w = client_weights[i][1] / total
                    global_weights[key] = global_weights[key] + w * client_weights[i][0][key] * (1-(client_dists[i][0] / total_client_dist))
        return global_weights, total
    
    def project(self, vec: torch.Tensor) -> np.ndarray:
        return (vec @ self.R).numpy()

    def train(self, dataset: Dataset, name_suffix: str='', write_logs: bool=True) -> str:
        
        name_suffix = '_' + name_suffix if name_suffix else ''
        run_id = time.asctime().replace(" ", "_").replace(":", "-")
        
        if write_logs:
            os.makedirs(f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}", exist_ok=True)

        for c in self.clients:
            for attack_type in attacks.attacks_to_prepare:
                if c.attack is not None and isinstance(c.attack, attack_type):
                    c.attack.prepare(self.model.state_dict()) # pyright: ignore[reportArgumentType]

        dataset_ref = ray.put(dataset)
        batch_size = ray.put(config.BATCH_SIZE)
        device = ray.put(config.DEVICE)

        if write_logs:
            # round 0:
            res = RoundResults(
                0,
                [],
                [],
                [],
                self.project(stats.flatten(self.model.state_dict())), # pyright: ignore[reportArgumentType]
                []
            )

            with open(f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}/round_0.npy", "wb") as f:
                pickle.dump(asdict(res), f)

            del res

        for round_id in range(1, config.ROUNDS+1):
            log.info(f"{round_id}/{config.ROUNDS}: ")
            client_accs = self.round(round_id, run_id, dataset_ref, batch_size, device, name_suffix, write_logs)
                            # self.round writes updates to disk ^^
            # history.append(asdict(res))

            acc = stats.avg(client_accs)
            log.info(f"\tAccuracy: {acc*100:.2f}%")

        return f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}"


    def round(self, round_id, run_id, dataset_ref: Dataset, batch_size, device, name_suffix: str, write_logs: bool) -> list[float]:
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        global_sd: models.StateDict = self.model.state_dict() # pyright: ignore[reportAssignmentType]

        # global_sd_ref = ray.put(global_sd)

        futures = [c.train(dataset_ref, batch_size, device) for c in selected]
        local_sds__cid = ray.get(futures)
        local_sds__cid.sort(key=lambda x: x[1])

        local_sds: tuple[ModelState, ...]
        local_sds, cid = zip(*local_sds__cid)

        proj_updates: list[np.ndarray] = []

        # calc deltas of each local_update since last round

        # del global_sd_ref

        flat_global = stats.flatten(global_sd)

        if write_logs:
            for sd, _total in local_sds:
                flat_local = stats.flatten(sd)
                delta = (flat_local - flat_global).cpu()
                proj_updates.append(self.project(delta))

        # train locally
        # if False:
        def dist_func(client_model_state: ModelState) -> float:
            return torch.norm(flat_global - stats.flatten(client_model_state[0]), p=2).item()
        
        client_types__selected = [(c.client_type, c.cid) for c in selected]
        client_types__selected.sort(key=lambda x: x[1])

        client_types, selected_id = zip(*client_types__selected)
        client_types_: list[str] = list(client_types)
        selected_id_: list[int] = list(selected_id)

        distances_selected = [(dist_func(sds), cid) for sds, cid in local_sds__cid]
        distances_not_selected = [(dist_func(c.model_state), c.cid) for c in self.clients if c.cid not in selected_id_] # pyright: ignore[reportArgumentType]
        distances = [i for i in sorted(distances_not_selected + distances_selected, key=lambda x: x[1])]
        total_dist = sum(i[1] for i in distances)

        # new_global = self.fed_avg_aggregate(local_sds)
        new_global = self.fed_attract_aggregate(local_sds, distances, total_dist)
        self.model.load_state_dict(new_global[0])

        ####################################################
        # Write the model to all clients

        # # fedavg
        # for c in self.clients:
        #     c.model_state = new_global[0]

        # fed_attract_aggregate
        for c in self.clients:
            # c.model_state = new_global[0]
            global_weights = new_global[0]

            if distances[c.cid][1] != c.cid:
                raise # for debugging, to see if distances are aligned

            dist_w = distances[c.cid][0]/total_dist

            for key in global_weights: # for each param in clients
                c.model_state[key] = global_weights[key] * (1-dist_w) + c.model_state[key] * dist_w
            # return global_weights, total

        ####################################################

        # for local_sd, cid in local_sds__cid:
        #     self.clients[cid].model_state = local_sd[0]

        if write_logs:
            log.debug("calcing stats.")

            res = RoundResults(
                round_id,
                selected_id_,
                [],
                proj_updates,
                self.project(stats.flatten(new_global[0])),
                client_types_
            )

            with open(f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}/round_{round_id}.npy", "wb") as f:
                pickle.dump(asdict(res), f)

            del res

        del proj_updates
        del local_sds
        gc.collect()

        # global_sd_ref = ray.put(new_global[0])

        log.debug("Finished training. Starting Eval")

        accs = ray.get([c.evaluate(dataset_ref, batch_size, device) for c in self.clients])

        accs.sort(key=lambda x: x[1])
        accs_, _ = zip(*accs)

        accs_lis = list(accs_)

        if write_logs:
            with open(f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}/round_{round_id}.npy", "rb") as f:
                data = pickle.load(f)

            data["client_accs"] = accs_lis

            with open(f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}/round_{round_id}.npy", "wb") as f:
                pickle.dump(data, f)

        return accs_lis
