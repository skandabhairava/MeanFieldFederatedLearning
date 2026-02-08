import ray
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import data
import models
import config
import attacks
from data import ClientSplit

class Client:
    def __init__(self, cid: int, splits: list[ClientSplit], client_type="normal"):
        self.cid = cid
        self.split = splits[cid]
        self.client_type = client_type
        # self.batch_size = batch_size

        if client_type == "byzantine_flip":
            self.attack = attacks.ByzantineFlip("flip")
        else:
            self.attack = None

    def train(self, global_sd: models.StateDict, dataset: Dataset, batch_size, device):
        return train.remote(global_sd, self.split, dataset, batch_size, device, self.attack)

    def evaluate(self, global_sd: models.StateDict, dataset: Dataset, batch_size, device):
        return evaluate.remote(global_sd, self.split, dataset, batch_size, device)
        # return evaluate(self.cid, global_sd, self.split, dataset, self.batch_size, config.DEVICE)

    @staticmethod    
    def sample_types(n, proportions: dict[str, float]):
        assert abs(1 - sum(proportions.values())) < 0.01, "Float values MUST add up to 1"

        names = list(proportions.keys())
        probs = list(proportions.values())
        return list(np.random.choice(names, size=n, p=probs))
    
@ray.remote(num_cpus=2, num_gpus=0.5)
def train(global_sd: models.StateDict, split: ClientSplit, dataset: Dataset, batch_size: int, device: str|torch.device, attack: None|attacks.Attack) -> models.StateDict:
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
            delta = new_sd[k] - global_sd[k]
            delta = attack.manipulate_update(delta)
            new_sd[k] = global_sd[k] + delta

    return new_sd # pyright: ignore[reportReturnType]

@ray.remote(num_cpus=2, num_gpus=0.5)
def evaluate(global_sd: models.StateDict, splits: ClientSplit, dataset: Dataset, batch_size: int, device: str|torch.device) -> float:
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

    return correct / total