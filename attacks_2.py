import torch
import models
import torch.nn.functional as F

import logging as log
import hashlib
import random
import copy

# log = logging.getLogger("attacks")

class Attack:
    def __init__(self, name):
        self.name = name
        self.prepared = False

    def manipulate_update(self, update: torch.Tensor, global_layer: torch.Tensor, param_name: str|None=None) -> torch.Tensor:
        return update
    
    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module]) -> None:
        self.prepared = True

class SubtleALIEAttack(Attack):
    layer_names_affected = []

    def __init__(self, name: str, epsilon: float = 0.001):
        super().__init__(name)
        self.epsilon = epsilon
        self.global_sd = {}
        self.shared_seed = None      # will be set in prepare
        self.sample = 0.3       # used only during prepare (read‑only later)
        self.copy_layer_names_affected = []

    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module]):
        """
        Called once per malicious client before any training.
        We use a fixed seed derived from the attack name (or a pre‑shared value)
        so that all colluding clients generate the same sign pattern.
        """
        super().prepare(ref_state_dict, model)
        # Generate a shared, deterministic seed from the attack name.
        # In practice this could be a pre‑agreed integer.
        seed_int = int(hashlib.md5(self.name.encode()).hexdigest()[:8], 16)
        self.shared_seed = seed_int

        if len(SubtleALIEAttack.layer_names_affected) == 0:
            # random.choices(list(ref_state_dict.keys()))
            layers = list(ref_state_dict.keys())
            SubtleALIEAttack.layer_names_affected = random.sample(layers, int(self.sample * len(layers)))


        self.copy_layer_names_affected = copy.deepcopy(SubtleALIEAttack.layer_names_affected)
        # log.info(f"{self.copy_layer_names_affected}")

    # def __getstate__(self):
    #     print("SERIALIZING:", self.copy_layer_names_affected)
    #     return self.__dict__

    # def __setstate__(self, state):
    #     print("DESERIALIZING:", state.get("copy_layer_names_affected"))
    #     self.__dict__.update(state)

    def manipulate_update(self,
                          update: torch.Tensor,
                          global_layer: torch.Tensor,
                          param_name: str|None = None) -> torch.Tensor:
        """
        Apply the subtle ALIE perturbation.

        - Sign is derived from a hash of (shared_seed + param_name).
        - Magnitude = epsilon * ||global_layer|| (small relative change).
        - The perturbation is added to the original update.
        """
        if not self.prepared :
            raise Exception()

        if param_name is None:
            raise ValueError("param_name must be provided for coordinated sign computation")
        
        if param_name not in self.copy_layer_names_affected:
            return update
        
        # print(f"ATTACK DEBUG: {param_name=}")
        # raise Exception(f"{param_name=}")

        # Deterministic sign shared by all malicious clients for this layer.
        sign_seed = f"{self.shared_seed}_{param_name}"
        sign_hash = int(hashlib.md5(sign_seed.encode()).hexdigest()[:8], 16)
        sign = 1.0 if (sign_hash % 2) == 0 else -1.0

        # Subtle magnitude: small fraction of the global layer's norm.
        # Using the global layer's norm ensures the perturbation is
        # proportional to the parameter scale, blending in naturally.
        global_norm = torch.norm(global_layer).item()
        if global_norm == 0:
            magnitude = self.epsilon
        else:
            magnitude = self.epsilon * global_norm

        # Apply perturbation.
        perturbation = sign * magnitude
        manipulated_update = update + perturbation

        return manipulated_update
    
#################################################################

class CleanLabelPoisoningAttack(Attack):
    """Clean‑label poisoning – adds a gradient from a trigger sample to the update."""
    def __init__(self, name: str, target_class: int = 0, trigger_pattern: str|None = None,
                 strength: float = 1.0, input_shape=(3, 32, 32)):
        super().__init__(name)
        self.target_class = target_class
        self.trigger_pattern = trigger_pattern   # currently unused, kept for future extension
        self.strength = strength
        self.input_shape = input_shape
        self.poison_grads = None

    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module]):
        # Build a model and load the global state
        super().prepare(ref_state_dict, model)
        model_ins = model()
        model_ins.eval()
        model_ins.load_state_dict(ref_state_dict, strict=False)
        model_ins.to('cpu')

        # Create a dummy input with a trigger (white square in bottom‑right corner)
        C, H, W = self.input_shape
        dummy = torch.zeros(1, C, H, W)
        # Place a 3x3 white square in the bottom‑right corner (adjust if H,W < 3)
        if H >= 3 and W >= 3:
            dummy[:, :, H-3:H, W-3:W] = 1.0

        target = torch.tensor([self.target_class])

        # Compute gradient of loss w.r.t. model parameters
        model_ins.zero_grad()
        output = model_ins(dummy)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Store the gradients using the same keys as the state dict
        self.poison_grads = {}
        for name, param in model_ins.named_parameters():
            if param.grad is not None:
                self.poison_grads[name] = param.grad.clone()

    def manipulate_update(self, update: torch.Tensor, global_layer: torch.Tensor, param_name: str|None = None) -> torch.Tensor:
        if not self.prepared :
            raise Exception()

        if self.poison_grads is None or param_name not in self.poison_grads:
            return update
        poison = self.poison_grads[param_name].to(update.device)
        return update + self.strength * poison

class DistributedCleanLabelPoisoningAttack(CleanLabelPoisoningAttack):
    """Distributed version – all attackers share the same target class and strength."""
    _shared_target = None
    _shared_trigger = None
    _shared_strength = None

    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module]):
        # Set shared variables once
        if DistributedCleanLabelPoisoningAttack._shared_target is None:
            DistributedCleanLabelPoisoningAttack._shared_target = self.target_class
            DistributedCleanLabelPoisoningAttack._shared_trigger = self.trigger_pattern
            DistributedCleanLabelPoisoningAttack._shared_strength = self.strength
        self.target_class = DistributedCleanLabelPoisoningAttack._shared_target
        self.trigger_pattern = DistributedCleanLabelPoisoningAttack._shared_trigger
        self.strength = DistributedCleanLabelPoisoningAttack._shared_strength
        super().prepare(ref_state_dict, model)

##########################################################################

class EdgeCasePoisoningAttack(Attack):
    """Edge‑case poisoning – scales the client’s update by a constant factor."""
    def __init__(self, name: str, scale: float = 2.0):
        super().__init__(name)
        self.prepared = True
        self.scale = scale

    def manipulate_update(self, update: torch.Tensor, global_layer: torch.Tensor, param_name: str|None = None) -> torch.Tensor:
        if not self.prepared :
            raise Exception()
        return update * self.scale

class DistributedEdgeCasePoisoningAttack(EdgeCasePoisoningAttack):
    """Distributed version – all attackers use the same scale."""
    _shared_scale = None

    def prepare(self, ref_state_dict: models.StateDict, model: type[torch.nn.Module]):
        if DistributedEdgeCasePoisoningAttack._shared_scale is None:
            DistributedEdgeCasePoisoningAttack._shared_scale = self.scale
        self.scale = DistributedEdgeCasePoisoningAttack._shared_scale
        super().prepare(ref_state_dict, model)

attacks_to_prepare: list[type[Attack]] = [
    SubtleALIEAttack,
    CleanLabelPoisoningAttack, DistributedCleanLabelPoisoningAttack,
    EdgeCasePoisoningAttack, DistributedEdgeCasePoisoningAttack
]