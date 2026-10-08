import ray
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from collections import Counter

import data
import models
import config
import attacks
from data import ClientSplit
from client_types import ClientTypes
import stats

from typing import Callable
import logging as log

ModelState = tuple[models.StateDict, int]

class Client:
    def __init__(
            self, 
            cid: int, 
            splits: list[ClientSplit], 
            proj_func: Callable[[torch.Tensor], torch.Tensor], 
            model_state: models.StateDict, 
            spill_folder: str,
            client_type: ClientTypes=ClientTypes.NORMAL, 
            save_log: bool = True,
            verifying_so_remove_splits: bool = False,
            seed=42
        ):
        self.cid = cid
        self.split = ([], []) if verifying_so_remove_splits else splits[cid]
        self.client_type: ClientTypes = client_type
        # self.model_state: StateDictPtr = StateDictPtr(model_state, project_func)
        self.model_state: None|models.StateDict = copy.deepcopy(model_state)
        self.model_state_flattened: None|torch.Tensor = proj_func(stats.flatten(self.model_state))
        # self.disp = proj_func(stats.flatten(self.model_state), disp=True) # pyright: ignore[reportCallIssue]
        self.in_memory_swapped: bool = True

        self.save_path = f"{self.cid}_model.pth"

        self.spill_folder = spill_folder

        self.attack: attacks.Attack|None = None

        if client_type == ClientTypes.SUBTLE:
            self.attack = attacks.SubtleAttack("subtle")
        elif client_type == ClientTypes.BACKDOOR_0:
            self.attack = attacks.BackdoorAttack("backdoor_0", 3, 0.3, 0)
        elif client_type == ClientTypes.BACKDOOR_0_ALL:
            self.attack = attacks.BackdoorAttack("backdoor_0_all", 3, 1, 0)
        elif client_type == ClientTypes.LABEL_SWITCH:
            self.attack = attacks.LabelSwitchAttack("label_switch", 10)
        elif client_type == ClientTypes.ALIE:
            self.attack = attacks.ALIEAttack("alie")

        if save_log:
            log.info(f"Client {self.cid} has been assigned type: {client_type}", extra={"save": True})

    def __repr__(self) -> str:
        return f"[{self.cid}: {self.client_type}]"

    def get_model_state(self) -> models.StateDict:
        if self.model_state is None:
            with open(f"{self.spill_folder}/{self.cid}.chkp", "rb") as f:
                state = torch.load(f)
        else:
            state = self.model_state
        return state

    def get_model_state_flattened(self) -> torch.Tensor:
        if self.model_state_flattened is None:
            with open(f"{self.spill_folder}/f{self.cid}.chkp", "rb") as f:
                state = torch.load(f)
        else:
            state = self.model_state_flattened
        return state

    def save_model_state(self):
        if self.model_state is not None:
            with open(f"{self.spill_folder}/{self.cid}.chkp", "wb") as f:
                torch.save(self.model_state, f)
    
    def save_model_state_flattened(self):
        if self.model_state_flattened is not None:
            with open(f"{self.spill_folder}/f{self.cid}.chkp", "wb") as f:
                torch.save(self.model_state_flattened, f)

    def clean_checkpoints(self):
        if os.path.exists(f"{self.spill_folder}/{self.cid}.chkp"):
            os.remove(f"{self.spill_folder}/{self.cid}.chkp")
        if os.path.exists(f"{self.spill_folder}/f{self.cid}.chkp"):
            os.remove(f"{self.spill_folder}/f{self.cid}.chkp")

    def train(self, dataset: Dataset, batch_size, device, round_id: int):
        # return train.remote(self.model_state.state_dict, self.split, dataset, batch_size, device, self.attack, self.cid)
        return train.remote(self.model_state, self.split, dataset, batch_size, device, self.attack, self.cid, round_id)

    def evaluate(self, dataset: Dataset, batch_size, device, should_attack: bool=False, calc_counts: bool=False):
        # return evaluate.remote(self.model_state.state_dict, self.split, dataset, batch_size, device, self.cid)
        return evaluate.remote(
            self.get_model_state(), 
            split, 
            dataset, 
            batch_size, 
            device, 
            self.attack if should_attack else None, 
            calc_counts,
            self.cid
        )

    def save(self, log_save_dir: str):
        with open(f"{log_save_dir}/{self.save_path}", "wb") as f:
            torch.save(self.get_model_state(), f)

    def _build_cluster_metadata_tree(self):
        return {
            'cid': self.cid,
            'type': "Client",
            'model_state_file': self.save_path,
            "attack": self.attack.name if self.attack is not None else None
        }

    @staticmethod
    def sample_types(n, proportions: dict[ClientTypes, float]) -> list[ClientTypes]:
        assert abs(1 - sum(proportions.values())) < 0.01, "Float values MUST add up to 1"

        names = list(proportions.keys())
        probs = list(proportions.values())

        return list(np.random.choice(names, size=n, p=probs))

@ray.remote(num_cpus=2, num_gpus=0.5)
def train(global_sd: models.StateDict, split: ClientSplit, dataset: Dataset, batch_size: int, device: str|torch.device, attack: None|attacks.Attack, cid: int, round_id: int) -> tuple[ModelState, int]:
    model = models.get_model().to(device)
    model.load_state_dict(global_sd)

    opt = optim.SGD(model.parameters(), lr=config.LR)
    model.train()

    # x: torch.Tensor
    # y: torch.Tensor

    train_loader = data.build_client_loaders(dataset, split, batch_size, True)
    for _ in range(config.LOCAL_EPOCHS):
        for x, y in train_loader:
            if attack is not None:
                x, y = attack.apply(x, y, modify_y=True)
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

    new_sd = model.cpu().state_dict()
    # new_sd = {k: v.cpu() for k, v in new_sd.items()}

    if attack is not None:
        # convert to update space, manipulate, then reconstruct
        for k in new_sd:
            delta: torch.Tensor = new_sd[k] - global_sd[k]            
            delta = attack.manipulate_update(delta, global_sd[k], param_name=k, round_id=round_id)
            new_sd[k] = global_sd[k] + delta

    return (new_sd, len(train_loader.dataset)), cid # pyright: ignore[reportArgumentType, reportReturnType]

@ray.remote(num_cpus=2, num_gpus=0.5)
def evaluate(
        global_sd: models.StateDict,
        splits: ClientSplit,
        dataset: Dataset,
        batch_size: int,
        device: str|torch.device,
        attack: None|attacks.Attack,
        calc_counts: bool,
        cid
    ) -> tuple[float, int, Counter[int]|None]:
    # x: torch.Tensor
    # y: torch.Tensor

    correct, total = 0, 0

    test_loader = data.build_client_loaders(dataset, splits, batch_size, False)

    # log.info(f"loaded test loader for client: {cid}")

    model = models.get_model().to(device)
    model.load_state_dict(global_sd)
    model.eval()

    if calc_counts:
        preds = []

    # print(f"{cid}: attack is applied: {attack is not None}")
    with torch.no_grad():
        # log.info(f"\tLoaded model for cid: {cid}")
        for i, (x, y) in enumerate(test_loader):
            if attack is not None:
                x, y = attack.apply(x, y, modify_y=False)
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            if calc_counts:
                preds.extend(pred.cpu().numpy().tolist()) # pyright: ignore[reportPossiblyUnboundVariable]

    counter = None
    if calc_counts:
        counter = Counter(preds) # pyright: ignore[reportPossiblyUnboundVariable]

    return (correct / total), cid, counter