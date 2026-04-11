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
            client_accs = self.round(round_id, run_id, dataset_ref, batch_size, device)

            acc = stats.avg(client_accs)
            log.info(f"\tAccuracy: {acc*100:.2f}% | {len(client_accs)} total clients evaluated.")
            # log.info(f"\tMean Dist from Center: {self.global_cluster.ema_mean} | Std: {self.global_cluster.ema_var**0.5}")
            self.global_cluster.print_tree()

        return f"{config.LOG_DIR}/RUN_{run_id}{name_suffix}"

    def round(self, round_id, run_id, dataset_ref: Dataset, batch_size, device) -> list[float]:
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