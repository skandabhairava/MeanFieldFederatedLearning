import copy

import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

import models
import attacks
import config
from data import ClientSplits
import data

class Client:
    def __init__(self, cid: int, splits: ClientSplits, combined_dataset: Dataset, batch_size: int, client_type="normal"):
        self.cid = cid
        self.splits = splits
        self.dataset = combined_dataset
        self.client_type = client_type
        self.batch_size = batch_size

        if client_type == "byzantine":
            self.attack = attacks.ByzantineFlip("flip")
        else:
            self.attack = None

    def train(self, global_sd: models.StateDict):
        return train.remote(self.cid, global_sd, self.splits, self.dataset, self.batch_size, config.DEVICE, self.attack)

    def evaluate(self, global_sd: models.StateDict):
        return evaluate.remote(self.cid, global_sd, self.splits, self.dataset, self.batch_size, config.DEVICE)

    @staticmethod    
    def sample_types(n, proportions: dict[str, float]):
        assert abs(1 - sum(proportions.values())) < 0.01, "Float values MUST add up to 1"

        names = list(proportions.keys())
        probs = list(proportions.values())
        return list(np.random.choice(names, size=n, p=probs))
    
@ray.remote(num_cpus=2, num_gpus=0.5)
def train(cid: int, global_sd: models.StateDict, splits: ClientSplits, dataset: Dataset, batch_size: int, device: str|torch.device, attack: None|attacks.Attack) -> models.StateDict:
    model = models.get_model().to(device)
    model.load_state_dict(global_sd)

    opt = optim.SGD(model.parameters(), lr=config.LR)
    model.train()

    # x: torch.Tensor
    # y: torch.Tensor

    train_loader = data.build_client_loaders(dataset, cid, splits, batch_size, True)
    for _ in range(config.LOCAL_EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

    new_sd = model.state_dict()

    if attack is not None:
        # convert to update space, manipulate, then reconstruct
        for k in new_sd:
            delta = new_sd[k] - global_sd[k]
            delta = attack.manipulate_update(delta)
            new_sd[k] = global_sd[k] + delta

    new_sd = {k: v.cpu() for k, v in new_sd.items()}

    return new_sd # pyright: ignore[reportReturnType]

@ray.remote(num_cpus=2, num_gpus=0.5)
def evaluate(cid: int, global_sd: models.StateDict, splits: ClientSplits, dataset: Dataset, batch_size: int, device: str|torch.device) -> float:
    # x: torch.Tensor
    # y: torch.Tensor

    correct, total = 0, 0

    test_loader = data.build_client_loaders(dataset, cid, splits, batch_size, False)

    model = models.get_model().to(device)
    model.load_state_dict(global_sd)
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total