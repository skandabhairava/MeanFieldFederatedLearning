import ray
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy

import data
import models
import config
import attacks
from data import ClientSplit
from client_types import ClientTypes
import stats

from typing import Callable

ModelState = tuple[models.StateDict, int]

class Client:
    def __init__(self, cid: int, splits: list[ClientSplit], proj_func: Callable[[torch.Tensor], torch.Tensor], model_state: models.StateDict, client_type: ClientTypes=ClientTypes.NORMAL, seed=42):
        self.cid = cid
        self.split = splits[cid]
        self.client_type: ClientTypes = client_type
        # self.model_state: StateDictPtr = StateDictPtr(model_state, project_func)
        self.model_state = copy.deepcopy(model_state)
        self.model_state_flattened = proj_func(stats.flatten(self.model_state))

        self.attack = None

        # if client_type == "byzantine_flip":
        #     self.attack = attacks.ByzantineFlip("flip")
        # elif client_type == "scaling_attack":
        #     self.attack = attacks.ScalingAttack("scaling")
        # elif client_type == "noise_injection_attack":
        #     self.attack = attacks.NoiseInjectionAttack("noise_injection")
        # elif client_type == "random_sign_attack":
        #     self.attack = attacks.RandomSignAttack("random_sign")
        # elif client_type == "norm_bound_attack":
        #     self.attack = attacks.NormBoundAttack("norm_bound")
        # elif client_type == "mean_shift_attack":
        #     self.attack = attacks.MeanShiftAttack("mean_shift")
        # elif client_type == "coordinated_krum_attack":
        #     self.attack = attacks.CoordinatedKrumAttack("coordinated_krum", seed=seed)
        # elif client_type == "sybil_attack":
        #     self.attack = attacks.SybilAttack("sybil_attack")
        # elif client_type == "sybil_attack2":
        #     self.attack = attacks.SybilAttack2("sybil_attack2")

    def train(self, dataset: Dataset, batch_size, device):
        # return train.remote(self.model_state.state_dict, self.split, dataset, batch_size, device, self.attack, self.cid)
        return train.remote(self.model_state, self.split, dataset, batch_size, device, self.attack, self.cid)

    def evaluate(self, dataset: Dataset, batch_size, device):
        # return evaluate.remote(self.model_state.state_dict, self.split, dataset, batch_size, device, self.cid)
        return evaluate.remote(self.model_state, self.split, dataset, batch_size, device, self.cid)

    @staticmethod    
    def sample_types(n, proportions: dict[ClientTypes, float]) -> list[str]:
        assert abs(1 - sum(proportions.values())) < 0.01, "Float values MUST add up to 1"

        names = list(proportions.keys())
        probs = list(proportions.values())
        return list(np.random.choice(names, size=n, p=probs))
    
    @staticmethod
    def propagate_downward():
        # should do nothing. Just exists to reduce run-time reflection check
        pass

@ray.remote(num_cpus=2, num_gpus=0.5)
def train(global_sd: models.StateDict, split: ClientSplit, dataset: Dataset, batch_size: int, device: str|torch.device, attack: None|attacks.Attack, cid: int) -> tuple[ModelState, int]:
    model = models.get_model().to(device)
    model.load_state_dict(global_sd)

    opt = optim.SGD(model.parameters(), lr=config.LR)
    model.train()

    # x: torch.Tensor
    # y: torch.Tensor

    train_loader = data.build_client_loaders(dataset, split, batch_size, True)
    for _ in range(config.LOCAL_EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

    new_sd = model.state_dict()
    new_sd = {k: v.cpu() for k, v in new_sd.items()}

    if attack is not None:
        # convert to update space, manipulate, then reconstruct
        for k in new_sd:
            delta: torch.Tensor = new_sd[k] - global_sd[k]            
            delta = attack.manipulate_update(delta, global_sd[k], param_name=k)
            new_sd[k] = global_sd[k] + delta

    return (new_sd, len(train_loader.dataset)), cid # pyright: ignore[reportArgumentType, reportReturnType]

@ray.remote(num_cpus=2, num_gpus=0.5)
def evaluate(global_sd: models.StateDict, splits: ClientSplit, dataset: Dataset, batch_size: int, device: str|torch.device, cid) -> tuple[float, int]:
    # x: torch.Tensor
    # y: torch.Tensor

    correct, total = 0, 0

    test_loader = data.build_client_loaders(dataset, splits, batch_size, False)

    # log.info(f"loaded test loader for client: {cid}")

    model = models.get_model().to(device)
    model.load_state_dict(global_sd)
    model.eval()

    with torch.no_grad():
        # log.info(f"\tLoaded model for cid: {cid}")
        for i, (x, y) in enumerate(test_loader):
            # log.info(f"\t\t{i} for cid: {cid}")
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return (correct / total), cid