import random
import copy
import logging as log
from dataclasses import dataclass, asdict
import time
import os
import pickle

import torch
import ray
import numpy as np

import stats
import config
import client
import models

@dataclass
class RoundResults:
    round: int
    stats: dict[str, int|float]
    cosine: np.ndarray
    client_accs: list[float]
    proj_updates: list[np.ndarray]
    proj_weights: list[np.ndarray]

class Server:
    def __init__(self, model: torch.nn.Module, clients: list[client.Client], seed: int):
        self.model = model
        self.clients = clients

        self.proj_dim = 50

        log.debug("starting server...")
        D = sum(p.numel() for p in self.model.state_dict().values())

        log.debug("configuring generator")

        g = torch.Generator().manual_seed(seed)

        log.debug("selecting dims")
        self.R = torch.randn(D, self.proj_dim, generator=g) / (self.proj_dim ** 0.5)

        log.debug("finished initing server")

    # FedAVG
    def aggregate(self, states: list[models.StateDict]):
        new = copy.deepcopy(states[0])
        for k in new:
            for i in range(1, len(states)):
                new[k] += states[i][k]
            new[k] /= len(states)
        return new
    
    def project(self, vec: torch.Tensor) -> np.ndarray:
        return (vec @ self.R).numpy()

    def train(self):
        # history = []

        run_id = time.asctime().replace(" ", "_").replace(":", "-")

        os.makedirs(f"{config.LOG_DIR}/RUN_{run_id}", exist_ok=True)

        for r in range(config.ROUNDS):
            log.info(f"{r+1}/{config.ROUNDS}: ")
            res = self.round(r)
            # history.append(asdict(res))
            with open(f"{config.LOG_DIR}/RUN_{run_id}/round_{r}.npy", "wb") as f:
                pickle.dump(asdict(res), f)
            
            acc = sum(res.client_accs)/len(res.client_accs)
            log.info(f"\tAccuracy: {acc*100:.2f}%")

        # return history


    def round(self, r) -> RoundResults:
        m = int(len(self.clients) * config.CLIENT_FRAC)
        selected = random.sample(self.clients, m)

        global_sd: models.StateDict = self.model.state_dict() # pyright: ignore[reportAssignmentType]

        futures = [c.train(global_sd) for c in selected]
        local_sds = ray.get(futures)

        raw_updates = []
        proj_weights: list[np.ndarray] = []
        proj_updates: list[np.ndarray] = []

        # calc deltas of each local_update since last round
        g = stats.flatten(global_sd)

        for sd in local_sds:
            flat_local = stats.flatten(sd)
            delta = (flat_local - g).cpu()
            
            raw_updates.append(delta)
            proj_updates.append(self.project(delta))
            proj_weights.append(self.project(flat_local))

        log.info("calcing stats.")

        st = stats.mean_var(raw_updates)
        cos = stats.cosine(raw_updates)

        new_global = self.aggregate(local_sds)
        self.model.load_state_dict(new_global)

        log.info("Finished training. Starting Eval")

        accs = ray.get([c.evaluate(new_global) for c in self.clients])

        log.info("finished. returning round data")

        return RoundResults(
            r,
            st,
            cos,
            accs,
            proj_updates,
            proj_weights
        )
