import torch
import logging as log
import copy
import numpy as np
from typing import Callable, Sequence
import models
import client
import stats
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
# from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
import torch.nn.functional as F
from uuid import uuid4
import math

from collections import OrderedDict
import config

class Cluster:
    def __init__(
            self,
            members: dict[int, 'client.Client|Cluster'],
            dist_func: Callable[[torch.Tensor, torch.Tensor], float],
            project_func: Callable[[torch.Tensor, torch.Tensor|None, bool, bool], torch.Tensor],
            parent: 'Cluster|None' = None,
            cluster_center: models.StateDict|None = None,
        ) -> None:
        self.members = OrderedDict(members)
        self.parent = parent
        self.dist_func = dist_func
        self.project_func = project_func
        self.cid = uuid4().int

        # Store flattened center
        if cluster_center is None:
            # Compute from members: average of client states or child cluster centers
            self.model_state = Cluster.avg_model_states([(child.cid, child.model_state) for child in members.values()])
        else:
            self.model_state = cluster_center
        self.model_state_flattened = self.project_func(
            stats.flatten(self.model_state),
            None,
            False,
            False
        )

        # distances and total_dist will be recomputed during updates
        self.dists, self.total_dist = self.calc_distances({cid: (c.model_state, c.model_state_flattened) for cid, c in self.members.items()}) #dict[int, tuple[models.StateDict, torch.Tensor]]
        self.has_split = False

    @staticmethod
    def hdbscan_features(
        proj: torch.Tensor,
        proj_dim: int,
        lambda_: float = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        proj: (4*proj_dim,)
            [pos | diff1 | diff2 | diff3]

        returns:
            [scaled_pos | lambda * normalized_history]
        """

        if not config.USE_VELOCITY:
            return proj

        pos = proj[:proj_dim]
        history = proj[proj_dim:]

        # Standardize each position dimension
        pos_mean = pos.mean(dim=0, keepdim=True)
        pos_std = pos.std(dim=0, keepdim=True).clamp_min(eps)
        pos = (pos - pos_mean) / pos_std

        # Normalize the ENTIRE history vector
        history = F.normalize(history, p=2, dim=0, eps=eps)

        return torch.cat(
            [
                pos,
                lambda_ * history,
            ]
        )

    def calc_distances(
            self, 
            member_states: dict[int, tuple[models.StateDict, torch.Tensor]]
        ) -> tuple[dict[int, float], float]:
        """
        Compute distances from each member's state to this cluster's flattened center.
        member_states: dict of (identifier: state_dict) where identifier is cid.
        """
        dists = {}
        total_dist = 0.0
        for ident, state__flat in member_states.items():
            m_state_flat = Cluster.hdbscan_features(self.model_state_flattened, 5)
            c_state_flat = Cluster.hdbscan_features(state__flat[1], 5)
            d = self.dist_func(c_state_flat, m_state_flat)
            dists[ident] = d
            total_dist += d
        return dists, total_dist

    def update_centers_upward(self, updated_client_states: dict[int, models.StateDict], use_softmax=False) -> int:
        """
        Recursively update cluster centers from leaves upward.
        updated_client_states: dict mapping client.cid -> new state_dict (after local training).
        """
        bigo_t = 0

        # First, update child clusters (if any) and collect their current states
        member_states: dict[int, tuple[models.StateDict, torch.Tensor]] = {}
        for cid, m in self.members.items():
            if isinstance(m, client.Client):                
                member_states[m.cid] = (m.model_state, m.model_state_flattened)
            else:
                # Recursively update child cluster
                bigo_t += m.update_centers_upward(updated_client_states, use_softmax=use_softmax)
                # After child cluster updated, its center is new
                member_states[m.cid] = m.model_state, m.model_state_flattened

        # Now compute distances for this cluster
        bigo_t += len(member_states)
        self.dists, self.total_dist = self.calc_distances(member_states)

        if use_softmax:
            self.model_state = Cluster.attract_aggregate_softmax(
                member_states, self.dists, self.model_state, self.total_dist
            )
            self.model_state_flattened = self.project_func(
                stats.flatten(self.model_state),
                self.model_state_flattened,
                True,
                False
            )
            return bigo_t

        self.model_state = Cluster.attract_aggregate(
            member_states, self.dists, self.model_state, self.total_dist
        )
        self.model_state_flattened = self.project_func(
            stats.flatten(self.model_state),
            self.model_state_flattened,
            True,
            False
        )
        return bigo_t

    def propagate_downward(self) -> int:
        """
        From root downward, update child clusters and clients using the new center and distances.
        """
        # First update this cluster's own members (clients or child clusters)
        inv_total = 1.0 / self.total_dist if self.total_dist > 0 else 0.0
        bigo_s = 1
        for cid, m in self.members.items():
            # Find distance for this member
            # Find matching dist
            dist_w = self.dists[m.cid] * inv_total
            one_minus_dist = (1 - dist_w)

            # For a client: update its model_state using convex combination
            if isinstance(m, client.Client):
                for key in self.model_state:
                    # m.model_state[key] = (self.model_state[key] * (1 - dist_w)) + (m.model_state[key] * dist_w)
                    m.model_state[key].mul_(dist_w).add_(self.model_state[key], alpha=one_minus_dist)

                m.model_state_flattened = self.project_func(
                    stats.flatten(m.model_state),
                    m.model_state_flattened,
                    False,
                    True
                )
            else:
                # For a child cluster: update its center and then propagate further down
                # Use same formula: child center = (parent_center*(1-dist_w)) + (child_center*dist_w)
                for key in self.model_state:
                    # m.model_state[key] = (self.model_state[key] * (1 - dist_w)) + (m.model_state[key] * dist_w)
                    m.model_state[key].mul_(dist_w).add_(self.model_state[key], alpha=one_minus_dist)

                m.model_state_flattened = self.project_func(
                    stats.flatten(m.model_state),
                    m.model_state_flattened,
                    False,
                    True
                )
                # Also update child's distances? Not needed; child will recompute when propagate_downward called on it.
                bigo_s += m.propagate_downward()

        return bigo_s

    def print_tree(self, level=0):
        indent = "  " * level
        if all(isinstance(m, client.Client) for m in self.members.values()):
            # Leaf cluster
            cids = [str(c.cid) for c in self.members.values()]
            log.info(f"{indent}Cluster (leaf) with clients: {', '.join(cids)}")
            # log.info(f"{indent}Outliers: {self.find_outlier_clients()}")
        else:
            log.info(f"{indent}Cluster (internal) with {len(self.members)} children")
            for m in self.members.values():
                if isinstance(m, Cluster):
                    m.print_tree(level+1)
                else:
                    log.info(f"{indent}  Client {m.cid}")

    @staticmethod
    def attract_aggregate(
        client_weights: dict[int, tuple[models.StateDict, torch.Tensor]],  # (state, ident, distance)
        dists: dict[int, float],
        cluster_center: models.StateDict,
        total_client_dist: float
    ) -> models.StateDict:
        """Weighted aggregation where weight = 1 - (dist/total_dist)."""
        if not client_weights:
            return copy.deepcopy(cluster_center)

        global_weights = copy.deepcopy(cluster_center)

        for key in global_weights.keys():
            total = 0.0
            assign = True

            for ident, state__flat in client_weights.items():
                w = (total_client_dist / dists[ident]) if dists[ident] != 0 else 1.0 #1 - (dists[ident] / total_client_dist) if total_client_dist > 0 else 1.0
                if assign:
                    global_weights[key] = state__flat[0][key] * w
                    assign = False
                else:
                    global_weights[key] += state__flat[0][key] * w

                total += w

            global_weights[key] /= total

        return global_weights

    @staticmethod
    def attract_aggregate_softmax(
        client_weights: dict[int, tuple[models.StateDict, torch.Tensor]],  # (state, ident, distance)
        dists: dict[int, float],
        cluster_center: models.StateDict,
        _total_client_dist: float
    ) -> models.StateDict:
        """Weighted aggregation where weight = 1 - (dist/total_dist)."""
        if not client_weights:
            return copy.deepcopy(cluster_center)

        T = 8.0 # temperature
        max_score = max(-d / T for d in dists.values())
        exp_scores = {ident: math.exp((-d/T) - max_score) for ident, d in dists.items()}
        sum_exp = sum(exp_scores.values())
        weights = {ident: exp / sum_exp for ident, exp in exp_scores.items()}

        global_weights = copy.deepcopy(cluster_center)
        for key in global_weights.keys():
            total = 0.0
            assign = True

            for ident, (state, _flat) in client_weights.items():
                w = weights[ident]
                if assign:
                    global_weights[key] = state[key] * w
                    assign = False
                else:
                    global_weights[key] += state[key] * w

                total += w

            global_weights[key] /= total

        return global_weights

    def split(self, assign_noise_to_nearest_cluster: bool=True) -> int:
        bigo_t = 0
        if self.has_split:
            for m in self.members.values():
                if isinstance(m, Cluster):
                    bigo_t += m.split(assign_noise_to_nearest_cluster)

            return bigo_t
        
        if len(self.members) == 2:
            log.info(f"Cluster '{self.cid}' has reached min_cluster_size")
            return bigo_t

        members: list['client.Client'] = []
        member_cid: list[int] = []

        for cid, mem in self.members.items():
            assert isinstance(mem, client.Client), f"{mem.cid} in cluster {self.cid} is supposed to be a CLIENT, not a CLuster."
            member_cid.append(cid)
            members.append(mem)

        X = np.array([Cluster.hdbscan_features(m.model_state_flattened, 5).numpy().astype(np.float64) for m in members])

        label_members: dict[int, list[int]] = {}

        for min_amt in range(min(len(self.members), 5), 1, -1):
            # dists = pairwise_distances(X).astype(np.float64)

            hdb = HDBSCAN(min_cluster_size=min_amt, copy=True, n_jobs=-1) # pyright: ignore[reportArgumentType]
            clusterer = hdb.fit(X)
            bigo_t += int(len(X) * math.log2(len(X)))

            # hdb = HDBSCAN(min_cluster_size=min_amt, copy=True, metric='precomputed', n_jobs=-1) # pyright: ignore[reportArgumentType]
            # clusterer = hdb.fit(dists)

            labels_all: np.ndarray = clusterer.labels_.copy()

            log.info(f"\t\t{labels_all=}")

            unique_clusters = set(labels_all) - {-1}

            # Find noise points
            noise_idx = np.where(labels_all == -1)[0]

            if len(noise_idx) != 0:
                unique_clusters_list = list(unique_clusters)
                if len(unique_clusters) > 1:
                    # Assign each noise point to nearest cluster center
                    centers = np.array([X[labels_all == c].mean(axis=0) for c in unique_clusters])
                    
                    nearest, _ = pairwise_distances_argmin_min(X[noise_idx], centers)
                    bigo_t += len(X[noise_idx])*len(centers)

                    for i, idx in enumerate(noise_idx):
                        labels_all[idx] = unique_clusters_list[nearest[i]]
                elif len(unique_clusters) == 1:
                    # Assign each noise point to THE ONLY cluster
                    for i, idx in enumerate(noise_idx):
                        labels_all[idx] = unique_clusters_list[0]

            label_members.clear()
            assert len(member_cid) == len(labels_all)
            for cid, l in zip(member_cid, labels_all):
                label_members.setdefault(l.item(), []).append(cid)

            if len(label_members) != 1:
                log.info(f"Cluster '{self.cid}' splitting with min cluster size: {min_amt}")
                break

        log.info(f"\t\t{label_members=}")
        if len(label_members) == 1:
            # if label_members is unique, it can't split
            log.info(f"Cluster '{self.cid}' children aren't diverse to form other clusters.")
            return bigo_t

        new_clusters_members_dict = [{idx: self.members[idx] for idx in cluster} for label_id, cluster in label_members.items()]
        new_clusters = [Cluster(i, self.dist_func, self.project_func, self) for i in new_clusters_members_dict]

        self.members = {c.cid: c for c in new_clusters}

        states = {cid: (c.model_state, c.model_state_flattened) for cid, c in self.members.items()}
        bigo_t += len(states)
        self.dists, self.total_dist = self.calc_distances(states)

        self.has_split = True

        return bigo_t

    def save_metadata(self, folder: str, file_name: str):
        tree = self._build_cluster_metadata_tree()
        with open(f"{folder}/{file_name}.top", "wb") as f:
            torch.save(tree, f)

    def _build_cluster_metadata_tree(self) -> dict[str, int|str|dict]:
        tree = {
            'cid': self.cid, 
            'type': "Cluster", 
            'model_state': self.model_state, 
            'members': {cid: member._build_cluster_metadata_tree() for cid, member in self.members.items()}
        }
        return tree

    @staticmethod
    def avg_model_states(client_weights: Sequence[tuple[int, models.StateDict]]) -> models.StateDict:
        if not client_weights:
            return {} # pyright: ignore[reportReturnType]
        
        global_weights = copy.deepcopy(client_weights[0][1]) # Copy 1st state dict
        for key in global_weights:
            global_weights[key] = client_weights[0][1][key].clone()
            for i in range(1, len(client_weights)):
                global_weights[key] += client_weights[i][1][key]

            global_weights[key] /= len(client_weights)
        return global_weights
