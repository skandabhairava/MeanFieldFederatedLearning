import torch
import models

import logging

log = logging.getLogger("attacks")

class Attack:
    def __init__(self, name):
        self.name = name
        self.prepared = False

    def manipulate_update(self, update: torch.Tensor, param_name=None, **kwargs):
        return update
    
    def prepare(self, ref_state_dict: models.StateDict) -> None:
        self.prepared = True
        pass


class ByzantineFlip(Attack):
    def manipulate_update(self, update, param_name=None, **kwargs):
        return -update


class ScalingAttack(Attack):
    def __init__(self, name, factor=10):
        super().__init__(name)
        self.factor = factor

    def manipulate_update(self, update, param_name=None, **kwargs):
        return update * self.factor
    

class NoiseInjectionAttack(Attack):
    def __init__(self, name, sigma=0.05):
        super().__init__(name)
        self.sigma = sigma

    def manipulate_update(self, update, param_name=None, **kwargs):
        noise = torch.randn_like(update) * self.sigma
        return update + noise
    

class RandomSignAttack(Attack):
    def manipulate_update(self, update, param_name=None, **kwargs):
        signs = torch.randint_like(update, low=0, high=2).float()
        signs = signs * 2 - 1
        return update * signs
    

class NormBoundAttack(Attack):
    def __init__(self, name, max_norm=5):
        super().__init__(name)
        self.max_norm = max_norm

    def manipulate_update(self, update, param_name=None, **kwargs):
        norm = torch.linalg.vector_norm(update)

        if norm == 0:
            return update

        return update / norm * self.max_norm
    
class MeanShiftAttack(Attack):
    def __init__(self, name, shift=3):
        super().__init__(name)
        self.shift = shift

    def manipulate_update(self, update, param_name=None, **kwargs):
        return update + self.shift

class LayerBackdoorAttack(Attack):
    def __init__(self, name, target_layer, strength=5):
        super().__init__(name)
        self.target_layer = target_layer
        self.strength = strength

    def manipulate_update(self, update, param_name: str|None=None, **kwargs):

        if param_name and self.target_layer in param_name:
            return update + torch.ones_like(update) * self.strength

        return update
    


class LayerBackdoorAttack2(Attack):
    def __init__(self, name: str, target_layer: str, strength: float = 2.0, stealth: float = 0.2):
        super().__init__(name)
        self.target_layer = target_layer
        self.strength = strength
        self.stealth = stealth
        self._trigger_direction: dict[str, torch.Tensor] = {}  # will be filled in prepare()

    def prepare(self, ref_state_dict: models.StateDict) -> None:
        super().prepare(ref_state_dict)

        for name, ref_tensor in ref_state_dict.items():
            if self.target_layer in name:
                direction = torch.randn_like(ref_tensor)
                norm = torch.linalg.vector_norm(direction)
                if norm > 0:
                    direction = direction / norm
                self._trigger_direction[name] = direction

    def manipulate_update(self, update: torch.Tensor, param_name: str|None = None, **kwargs) -> torch.Tensor:

        assert self.prepared, f"'{self.name}' ATTACK HASN'T BEEN PREPARED"

        if param_name is None or param_name not in self._trigger_direction:
            return update

        trigger = self._trigger_direction[param_name]
        poisoned = update + self.strength * trigger

        # Preserve the original norm
        orig_norm = torch.linalg.vector_norm(update)
        new_norm = torch.linalg.vector_norm(poisoned)
        if new_norm > 0:
            poisoned = poisoned * (orig_norm / new_norm)

        # Add stealth noise (scaled by the standard deviation of the honest update)
        std = torch.std(update) if update.numel() > 1 else 0.0
        if std > 0:
            stealth_noise = torch.randn_like(update) * self.stealth * std
        else:
            stealth_noise = torch.zeros_like(update)
        return poisoned + stealth_noise    


class CoordinatedKrumAttack(Attack):
    def __init__(self, name: str, strength: float = 3.0, seed: int = 42):
        super().__init__(name)
        self.strength = strength
        self.seed = seed
        self._direction: dict[str, torch.Tensor] = {}  # will be filled in prepare()

    def prepare(self, ref_state_dict: models.StateDict) -> None:
        super().prepare(ref_state_dict)

        for name, ref_tensor in ref_state_dict.items():
            generator = torch.Generator(device=ref_tensor.device)

            # Combine seed with parameter name hash to get per‑parameter deterministic direction
            generator.manual_seed(self.seed + hash(name) % 2**32)
            direction = torch.randn_like(ref_tensor, generator=generator)
            norm = torch.linalg.vector_norm(direction)

            if norm > 0:
                direction = direction / norm
            self._direction[name] = direction

    def manipulate_update(self, update: torch.Tensor, param_name: str|None = None, **kwargs) -> torch.Tensor:

        assert self.prepared, f"'{self.name}' ATTACK HASN'T BEEN PREPARED"

        if param_name is None or param_name not in self._direction:
            return update
        direction = self._direction[param_name]
        return update + self.strength * direction
    

class SybilAttack(Attack):
    def __init__(self, name: str, strength: float = 4.0):
        super().__init__(name)
        self.strength = strength
        self._direction: dict[str, torch.Tensor] = {}

    def prepare(self, ref_state_dict: models.StateDict) -> None:
        super().prepare(ref_state_dict)

        for name, ref_tensor in ref_state_dict.items():
            direction = torch.randn_like(ref_tensor)
            norm = torch.linalg.vector_norm(direction)
            if norm > 0:
                direction = direction / norm
            self._direction[name] = direction

    def manipulate_update(self, update: torch.Tensor, param_name: str|None = None, **kwargs) -> torch.Tensor:

        assert self.prepared, f"'{self.name}' ATTACK HASN'T BEEN PREPARED"

        if param_name is None or param_name not in self._direction:
            return update
        direction = self._direction[param_name]
        return update + self.strength * direction
    

class SybilAttack2(Attack):
    def __init__(self, name: str, correlation: float = 0.9, strength: float = 3.0, client_id: int = 0):
        super().__init__(name)
        self.correlation = correlation
        self.strength = strength
        self.client_id = client_id
        self._base_direction: dict[str, torch.Tensor] = {}   # common base (shared across clients)
        self._client_direction: dict[str, torch.Tensor] = {} # per‑client direction

    def prepare(self, ref_state_dict: models.StateDict) -> None:
        super().prepare(ref_state_dict)

        for name, ref_tensor in ref_state_dict.items():
            # Common base direction (deterministic across all clients)
            base_gen = torch.Generator(device=ref_tensor.device)
            base_gen.manual_seed(999 + hash(name) % 2**32)  # fixed seed for base
            base = torch.randn_like(ref_tensor, generator=base_gen)
            norm_base = torch.linalg.vector_norm(base)
            if norm_base > 0:
                base = base / norm_base
            self._base_direction[name] = base

            # Per‑client direction
            client_gen = torch.Generator(device=ref_tensor.device)
            client_gen.manual_seed(self.client_id + hash(name) % 2**32)
            rand = torch.randn_like(ref_tensor, generator=client_gen)
            norm_rand = torch.linalg.vector_norm(rand)
            if norm_rand > 0:
                rand = rand / norm_rand
            self._client_direction[name] = rand

    def manipulate_update(self, update: torch.Tensor, param_name: str|None = None, **kwargs) -> torch.Tensor:

        assert self.prepared, f"'{self.name}' ATTACK HASN'T BEEN PREPARED"

        if param_name is None or param_name not in self._base_direction:
            return update

        base = self._base_direction[param_name]
        rand = self._client_direction[param_name]

        # Combine base and random
        direction = self.correlation * base + (1 - self.correlation) * rand
        norm_dir = torch.linalg.vector_norm(direction)
        if norm_dir > 0:
            direction = direction / norm_dir

        # Apply malicious shift and preserve original norm
        poisoned = update + self.strength * direction
        orig_norm = torch.linalg.vector_norm(update)
        new_norm = torch.linalg.vector_norm(poisoned)
        if new_norm > 0:
            poisoned = poisoned * (orig_norm / new_norm)

        return poisoned
    

attacks_to_prepare: list[type[Attack]] = [LayerBackdoorAttack2, CoordinatedKrumAttack, SybilAttack, SybilAttack2]