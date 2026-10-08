from collections import OrderedDict
import math

import torch
import models
import torch.nn.functional as F
from torch import optim

import hashlib
import data
import random
import copy
import config
from client_types import ClientTypes
import stats

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import client

# log = logging.getLogger("attacks")

class Attack:
    def __init__(self, name):
        self.name = name
        self.prepared = False

    def manipulate_update(self, update: torch.Tensor, global_layer: torch.Tensor, param_name: str|None=None, round_id: int|None=None) -> torch.Tensor:
        return update
    
    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module], dataset, split) -> None:
        self.prepared = True

    def apply(self, x: torch.Tensor, y: torch.Tensor, round_id: int, modify_y: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        return x, y

    def perform_general_post_update_attack(
        self,
        clients: list['client.Client'],
        n_total_in_round: int,
        n_byzantine_in_round: int,
        round_id: int,
    ) -> None:
        return None

class ModelReplace(Attack):
    def __init__(self, name, scale: float|int = 30):
        self.name = name
        self.prepared = False
        self.scale = scale

    def manipulate_update(self, update: torch.Tensor, global_layer: torch.Tensor, param_name: str|None=None, round_id: int|None=None) -> torch.Tensor:
        return update * self.scale

class Random(Attack):
    def __init__(self, name, start: int = 0):
        self.name = name
        self.prepared = False
        self.start = start

    def manipulate_update(
            self,
            update: torch.Tensor,
            global_layer: torch.Tensor,
            param_name: str|None = None,
            round_id: int|None=None
        ) -> torch.Tensor:
            """
            Apply the subtle perturbation.
            """
            if not self.prepared :
                raise Exception()
    
            if round_id is not None and round_id < self.start:
                return update

            return update + torch.zeros_like(update).normal_(mean=0.1, std=0.1)

class SignFlipping(Attack):
    def __init__(self, name, start: int = 0):
        self.name = name
        self.prepared = False
        self.start = start

    def manipulate_update(
            self,
            update: torch.Tensor,
            global_layer: torch.Tensor,
            param_name: str|None = None,
            round_id: int|None=None
        ) -> torch.Tensor:
            """
            Apply the subtle perturbation.
            """
            if not self.prepared :
                raise Exception()
    
            if round_id is not None and round_id < self.start:
                return update

            return -update

class ALIEAttack(Attack):
    global_model_state: None|models.StateDict = None
    def __init__(self, name, start: int=0):
        self.name = name
        self.prepared = False
        self.start = start

    @staticmethod
    def compute_z_max(n_total_in_round: int, n_byzantine_in_round: int) -> float:
        """
        Compute ALIE z_max using the inverse standard normal CDF.
        No scipy needed — uses torch.erfinv.
        """
        n = n_total_in_round
        m = n_byzantine_in_round
        n_benign = n - m

        if n_benign <= 0:
            return 5.0

        majority_threshold = math.floor(n / 2 + 1)
        s = majority_threshold - m

        if s <= 0:
            return 5.0

        # Probability threshold: (n - m - s) / (n - m)
        prob = (n - m - s) / (n - m)

        # Φ⁻¹(p) = √2 * erfinv(2p - 1)
        p_t = torch.tensor(prob, dtype=torch.float64)
        z_max = (math.sqrt(2) * torch.erfinv(2 * p_t - 1)).item()

        return z_max

    def perform_general_post_update_attack(
            self,
            clients: list['client.Client'],
            n_total_in_round: int,
            n_byzantine_in_round: int,
            round_id: int,
            z: float|None = None#0.2
        ):
        """
        Perform the ALIE (A Little Is Enough) attack on clients marked with client_type == ALIE.

        Args:
            clients: List of Client objects. Each client has attributes:
                    - cid: int
                    - model_state: OrderedDict (state dict of the model after local training)
                    - client_type: ClientTypes
            global_model_state: OrderedDict, the global model state before aggregation.
            z: The scaling factor for the standard deviation in the attack (default: 0.1).
        """

        if round_id < self.start:
            return

        assert ALIEAttack.global_model_state is not None, "Global model in ALIE attack is None"

        if z is None:
            z = ALIEAttack.compute_z_max(n_total_in_round, n_byzantine_in_round)

        # print(f"{n_byzantine_in_round=} | Attack {z=}")

        # Separate benign clients (non-ALIE) and malicious clients (ALIE)
        benign_clients = [c for c in clients if c.client_type != ClientTypes.ALIE and c.client_type != ClientTypes.LATE_ALIE]
        malicious_clients = [c for c in clients if c.client_type == ClientTypes.ALIE or c.client_type == ClientTypes.LATE_ALIE]

        if not benign_clients or not malicious_clients:
            # No benign clients to compute statistics, or no malicious clients to attack
            return

        # Compute the parameter-wise updates for benign clients
        # updates_list = [client.model_state - global_model_state] for each benign client
        benign_updates = []
        for client in benign_clients:
            update = OrderedDict()
            for key in ALIEAttack.global_model_state.keys():
                # Ensure both are on the same device (move to CPU for simplicity)
                global_param = ALIEAttack.global_model_state[key]#.cpu()
                client_param = client.get_model_state()[key]#.cpu()
                update[key] = client_param - global_param
            benign_updates.append(update)

        # Compute mean and standard deviation per parameter across benign updates
        # Initialize mean_dict and std_dict as zero tensors
        mean_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        std_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key in ALIEAttack.global_model_state.keys():
            # Stack all benign updates for this parameter (shape: [num_benign, *param_shape])
            stacked = torch.stack([upd[key] for upd in benign_updates], dim=0)
            mean_dict[key] = stacked.mean(dim=0)
            std_dict[key] = stacked.std(dim=0, unbiased=False)  # population std

        # For each malicious client, compute the poisoned update: mean - z * std
        # z_recreate = []

        for mal_client in malicious_clients:
            poisoned_update = OrderedDict()
            # z_avg = []
            for key in ALIEAttack.global_model_state.keys():
                # Compute malicious update for this parameter
                mask = std_dict[key] < 1e-5
                std_dict[key][mask] = 1e-5
                mal_update = mean_dict[key] - z * std_dict[key]
                # The client's new model state = global_model + malicious_update
                # Move to the same device as the original model_state if needed
                device = mal_client.get_model_state()[key].device
                poisoned_update[key] = (ALIEAttack.global_model_state[key] + mal_update).to(device)

            # Replace the client's model_state with the poisoned one
            mal_client.model_state = poisoned_update

        # shift0 = torch.norm(stats.flatten(malicious_clients[0].get_model_state()) - stats.flatten(ALIEAttack.global_model_state))
        # shift1 = torch.norm(stats.flatten(malicious_clients[1].get_model_state()) - stats.flatten(ALIEAttack.global_model_state))
        # shift2 = torch.norm(stats.flatten(malicious_clients[2].get_model_state()) - stats.flatten(ALIEAttack.global_model_state))

        # print(f"ALIE: z={z:.3f}, poisoning {len(malicious_clients)} clients, ||shift||={shift0:.4f}, {shift1:.4f}, {shift2:.4f}")
        # mu_n = torch.mean(torch.cat([m.flatten() for m in mean_dict.values()]))
        # sd_n = torch.mean(torch.cat([s.flatten() for s in std_dict.values()]))
        # print(f"||mu||={mu_n:.4f} ||sigma||={sd_n:.4f} z*||sigma||={z*sd_n:.4f} ratio={z*sd_n/mu_n:.3f}")

class IPMAttack(Attack):
    """
    Inner Product Manipulation (Xie, Koyejo, Gupta 2019, https://arxiv.org/abs/1903.03936).
    Byzantine clients submit  ref - epsilon * (mean(honest) - ref),
    where `ref` is the mean of honest clients' starting states, set by the server each round.
    """

    global_model_state: models.StateDict | None = None  # set by server each round

    def __init__(self, name: str = "ipm", start: int = 0):
        super().__init__(name)
        self.start = start
        # print(f"Starting IPM at: {start}")

    def prepare(self, ref_state_dict, model, dataset, split) -> None:
        super().prepare(ref_state_dict, model, dataset, split)
        IPMAttack.global_model_state = None

    def perform_general_post_update_attack(
        self,
        clients: list['client.Client'],
        n_total_in_round: int,
        n_byzantine_in_round: int,
        round_id: int
    ) -> None:

        if round_id < self.start:
            return

        ref = IPMAttack.global_model_state
        assert ref is not None

        honest, byz = [], []
        for c in clients:
            if c.model_state is None:
                continue  # didn't participate this round
            if c.client_type == ClientTypes.IPM or c.client_type == ClientTypes.LATE_IPM:
                byz.append(c)
            else:
                honest.append(c)
        if not honest or not byz:
            return

        epsilon = len(clients)

        ref_flat = stats.flatten(ref).float()
        honest_mean = torch.zeros_like(ref_flat)
        for c in honest:  # running sum, never stacks N copies
            honest_mean += stats.flatten(c.model_state).float().to(ref_flat.device)
        honest_mean /= len(honest)

        mal_sd = stats.unflatten(ref_flat - epsilon * (honest_mean - ref_flat), ref)

        for c in byz:
            new_sd = dict(c.model_state)  # keeps the attacker's own BN buffers
            for k, v in mal_sd.items():
                new_sd[k] = v.clone()
            c.model_state = new_sd

class SubtleAttack(Attack):
    layer_names_affected = []
    direction: dict[str, torch.Tensor] = {}  # will be filled in prepare()

    def __init__(self, name: str, sample: float = 1.0, start: int=0):
        super().__init__(name)
        # self.epsilon = epsilon
        self.global_sd = {}
        self.shared_seed = None      # will be set in prepare
        self.sample = sample       # used only during prepare (read‑only later)
        self.copy_layer_names_affected = []
        self.start = start

    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module], dataset, split):
        """
        Called once per malicious client before any training.
        We use a fixed seed derived from the attack name (or a pre-shared value)
        so that all colluding clients generate the same sign pattern.
        """
        super().prepare(ref_state_dict, model, dataset, split)
        # Generate a shared, deterministic seed from the attack name.
        # In practice this could be a pre‑agreed integer.
        seed_int = int(hashlib.md5(self.name.encode()).hexdigest()[:8], 16)
        self.shared_seed = seed_int

        if len(SubtleAttack.layer_names_affected) == 0 and self.sample != 1.0:
            # random.choices(list(ref_state_dict.keys()))
            layers = list(ref_state_dict.keys())
            SubtleAttack.layer_names_affected = random.sample(layers, int(self.sample * len(layers)))

        if len(SubtleAttack.direction) == 0:
            m = model()
            opt = optim.SGD(m.parameters(), lr=config.LR)
            m.train()
            
            # x: torch.Tensor
            # y: torch.Tensor
        
            old_sd = m.cpu().state_dict()
            m.to(config.DEVICE)

            train_loader = data.build_client_loaders(dataset, split, config.BATCH_SIZE, True)
            for _ in range(config.LOCAL_EPOCHS):
                for x, y in train_loader:
                    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        
                    opt.zero_grad()
                    loss = F.cross_entropy(m(x), y)
                    loss.backward()
                    opt.step()
            
            new_sd = m.cpu().state_dict()

            for name, new_tensor in new_sd.items():

                if self.sample != 1.0 and name not in SubtleAttack.layer_names_affected:
                    continue

                SubtleAttack.direction[name] = -15 * 2 * (new_tensor - old_sd[name])

        if self.sample != 1.0:
            self.copy_layer_names_affected = copy.deepcopy(SubtleAttack.layer_names_affected)

        self.copy_direction = copy.deepcopy(SubtleAttack.direction)
        # print(self.copy_direction)

    def manipulate_update(self,
                          update: torch.Tensor,
                          global_layer: torch.Tensor,
                          param_name: str|None = None,
                          round_id: int|None=None) -> torch.Tensor:
        """
        Apply the subtle perturbation.
        """
        if not self.prepared :
            raise Exception()

        if round_id is not None and round_id < self.start:
            return update

        if param_name is None:
            raise ValueError("param_name must be provided for coordinated sign computation")
        
        if self.sample != 1.0 and param_name not in self.copy_layer_names_affected:
            return update
        
        # print(f"ATTACK DEBUG: {param_name=}")
        # raise Exception(f"{param_name=}")

        # Deterministic sign shared by all malicious clients for this layer.
        # manipulated_update = update + self.copy_direction[param_name] * self.epsilon

        manipulated_update = self.copy_direction[param_name]# * 25 * update.norm()
        
        # dist = update.norm()
        # T_norm = 0.5
        # manipulated_update_size = (1 - ((5*T_norm)/(3*dist)))
        # manipulated_update = manipulated_update_size * update
        
        # if round_id == 1:
        # manipulated_update = -update

        # manipulated_update = torch.zeros_like(update)
        

        # eps = 1e-8
        # # manipulated_update_scaled = manipulated_update * (update.norm(p=2) / manipulated_update.norm(p=2) + eps)

        # # value = 1
        # final_value = update.norm(p=2)
        # final_value = final_value + 0.5*final_value

        # round_id = 10 if round_id is None else round_id
        # value = max(((round_id-1)*(final_value - 1))/(5) + 1, final_value)

        # print(dist, manipulated_update.norm(), manipulated_update_size, manipulated_update.size())

        # manipulated_update_scaled = manipulated_update * (value / manipulated_update.norm(p=2) + eps)

        return manipulated_update
    
#####################################################

class BackdoorAttack(Attack):
    def __init__(self, name: str, trigger_size: int = 3, poison_fraction: float = 0.3, target_class: int = 0, start: int=0):
        super().__init__(name)
        self.trigger_size = trigger_size
        self.poison_fraction = poison_fraction
        self.target_class = target_class
        self.trigger_pattern: None|torch.Tensor = None   # will be created in prepare()
        self.start = start

    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module], dataset, split) -> None:
        """Pre-compute the trigger pattern (a white square) once."""
        # Assuming images are in [0,1] range. For [0,255] use 255.0 instead of 1.0.
        self.trigger_pattern = torch.full((3, self.trigger_size, self.trigger_size), 1.0)
        self.prepared = True

    def apply(self, x: torch.Tensor, y: torch.Tensor, round_id: int, modify_y: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        """Add trigger to a random subset of the batch and set their labels to target_class."""

        if not self.prepared or not isinstance(self.trigger_pattern, torch.Tensor):
            raise Exception()

        if round_id < self.start:
            return x, y

        batch_size = x.size(0)
        # Decide which samples to poison
        poison_mask = torch.rand(batch_size) < 0.3 #self.poison_fraction

        if poison_mask.any():
            # Add the trigger pattern to the bottom‑right corner of poisoned images
            x_poisoned = x[poison_mask]
            # Ensure trigger pattern is on the same device and broadcastable
            trigger = self.trigger_pattern.to(x.device)
            # Place trigger at bottom‑right corner
            x_poisoned[:, :, -self.trigger_size:, -self.trigger_size:] = trigger
            x[poison_mask] = x_poisoned

            if modify_y:
                y[poison_mask] = self.target_class

        return x, y


class LabelSwitchAttack(Attack):
    label_switch = {}

    def __init__(self, name: str, total_labels: int, start: int=0):
        super().__init__(name)
        self.total_labels = total_labels
        self.start = start

    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module], dataset, split) -> None:
        """Pre-compute the trigger pattern (a white square) once."""
        # Assuming images are in [0,1] range. For [0,255] use 255.0 instead of 1.0.
        labels = list(range(self.total_labels))
        if len(LabelSwitchAttack.label_switch) == 0:
            for label, rnd_label in zip(labels, random.sample(labels, k=len(labels))):
                LabelSwitchAttack.label_switch[label] = rnd_label
        self.prepared = True

    def apply(self, x: torch.Tensor, y: torch.Tensor, round_id: int, modify_y: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        """Add trigger to a random subset of the batch and set their labels to target_class."""

        if not self.prepared:
            raise Exception()

        if round_id < self.start:
            return x, y

        # x = torch.tensor([1, 2, 5, 3])

        lut = torch.arange(self.total_labels)
        for old, new in LabelSwitchAttack.label_switch.items():
            lut[old] = new

        y = lut[y]

        return x, y