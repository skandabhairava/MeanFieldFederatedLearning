import os

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
from client_types import ClientTypes

from collections import OrderedDict
import config

class Cluster:
    def __init__(
            self,
            members: dict[int, 'client.Client|Cluster'],
            dist_func: Callable[[torch.Tensor, torch.Tensor, bool, float|None], float],
            project_func: Callable[[torch.Tensor, torch.Tensor|None, bool, bool, bool], torch.Tensor],
            spill_folder: str,
            in_memory_swapped: bool = True,
            parent: 'Cluster|None' = None,
            cluster_center: models.StateDict|None = None,
        ) -> None:
        self.members = OrderedDict(members)
        self.parent = parent
        self.dist_func = dist_func
        self.project_func = project_func
        self.cid = uuid4().int
        self.spill_folder = spill_folder

        self.in_memory_swapped: bool = in_memory_swapped
        self.model_state: None|models.StateDict

        # Store flattened center
        if cluster_center is None:
            # Compute from members: average of client states or child cluster centers
            self.model_state = Cluster.avg_model_states(
                [
                    (child.cid, child.get_model_state()) 
                    for child in members.values()
                ])
        else:
            self.model_state = cluster_center


        self.model_state_flattened: None|torch.Tensor = self.project_func(
            stats.flatten(self.model_state),
            None,
            False,
            False,
            False
        )

        # distances and total_dist will be recomputed during updates
        self.dists, self.total_dist = self.calc_distances(
            {
                cid: (c.get_model_state(), c.get_model_state_flattened()) 
                for cid, c in self.members.items()
            }) #dict[int, tuple[models.StateDict, torch.Tensor]]
        self.has_split = False

    def new_subsplit(self, dists: dict[int, float], members: dict[int, 'client.Client|Cluster']):
        member_states = {}
        dists2 = {}
        for m in members.values():
            member_states[m.cid] = m.get_model_state(), m.get_model_state_flattened()
            dists2[m.cid] = dists[m.cid]
        center = Cluster.attract_aggregate(member_states, dists2, self.get_model_state(), sum(dists2.values()), self.cid)

        return Cluster(
            members,
            self.dist_func,
            self.project_func,
            self.spill_folder,
            self.in_memory_swapped,
            self,
            center
        )
        

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
        max_dist = float('-inf')
        for ident, state__flat in member_states.items():
            # Cluster.hdbscan_features(x) returns x, when use velocity is disabled in config
            # m_state_flat = Cluster.hdbscan_features(self.get_model_state_flattened(), 5)
            # c_state_flat = Cluster.hdbscan_features(state__flat[1], 5)

            m_state_flat = self.get_model_state_flattened()
            c_state_flat = state__flat[1]

            d = self.dist_func(c_state_flat, m_state_flat, False, None)
            # print(d, c_state_flat[:2], m_state_flat[:2])
            dists[ident] = d
            total_dist += d
            max_dist = max(d, max_dist)

        if not config.USE_VELOCITY:
            return dists, total_dist

        # median = np.quantile(list(dists.values()), q=0.5)
        
        # max_dist2 = float('-inf')
        # total_dist = 0
        # for ident, state__flat in member_states.items():
        #     # Cluster.hdbscan_features(x) returns x, when use velocity is disabled in config
        #     # m_state_flat = Cluster.hdbscan_features(self.get_model_state_flattened(), 5)
        #     # c_state_flat = Cluster.hdbscan_features(state__flat[1], 5)

        #     m_state_flat = self.get_model_state_flattened()
        #     c_state_flat = state__flat[1]

            
        #     sim1 = (1 - torch.cosine_similarity(m_state_flat[10:20], c_state_flat[10:20], dim=0))/2
        #     sim2 = (1 - torch.cosine_similarity(m_state_flat[20:30], c_state_flat[20:30], dim=0))/2
        #     sim3 = (1 - torch.cosine_similarity(m_state_flat[30:40], c_state_flat[30:40], dim=0))/2

        #     d = ((1 - math.e**(-dists[ident]/median)) + 3*(sim1 + sim2 + sim3))/10

        #     # d = (1 - torch.cosine_similarity(m_state_flat[10:], c_state_flat[10:], dim=0))/2

        #     dists[ident] = d
        #     total_dist += d
        #     max_dist2 = max(d, max_dist2)
        return dists, total_dist

    def update_centers_upward(self, bigo_W_size: int, bigo_W_flat_size: int) -> int:
        """
        Recursively update cluster centers from leaves upward.
        updated_client_states: dict mapping client.cid -> new state_dict (after local training).
        """
        bigo_t = 0

        # First, update child clusters (if any) and collect their current states
        member_states: dict[int, tuple[models.StateDict, torch.Tensor]] = {}
        for cid, m in self.members.items():
            if isinstance(m, client.Client):                
                member_states[m.cid] = (m.get_model_state(), m.get_model_state_flattened())
            else:
                # Recursively update child cluster
                bigo_t += m.update_centers_upward(bigo_W_size, bigo_W_flat_size)
                # After child cluster updated, its center is new
                member_states[m.cid] = m.get_model_state(), m.get_model_state_flattened()

        # Now compute distances for this cluster
        bigo_t += len(member_states)*bigo_W_flat_size
        self.dists, self.total_dist = self.calc_distances(member_states)

        # print(f"{self.cid} after training dists: \n\t{'\n\t'.join([f'{i}: {d}' for i, d in self.dists.items()])}")

        bigo_t += len(member_states)*bigo_W_size
        self.model_state = Cluster.attract_aggregate(
            member_states, self.dists, self.get_model_state(), self.total_dist, self.cid
        )

        bigo_t += bigo_W_size*bigo_W_flat_size
        self.model_state_flattened = self.project_func(
            stats.flatten(self.model_state),
            self.model_state_flattened,
            True,
            False,
            False
        )

        if not self.in_memory_swapped:
            self.save_model_state()
            self.save_model_state_flattened()
            self.model_state = None
            self.model_state_flattened = None

        # print(self.cid, stats.flatten(self.model_state)[:2])

        return bigo_t

    def propagate_downward(self, recalc_dists: bool, bigo_W_size: int, bigo_W_flat_size: int) -> tuple[int, int]:
        """
        From root downward, update child clusters and clients using the new center and distances.
        """
        # First update this cluster's own members (clients or child clusters)
        bigo_s = bigo_W_size + bigo_W_flat_size
        bigo_t = 0

        if recalc_dists:
            member_states: dict[int, tuple[models.StateDict, torch.Tensor]] = {
                mcid: (m.get_model_state(), m.get_model_state_flattened()) 
                for mcid, m in self.members.items()
            }

            bigo_t += len(member_states)*bigo_W_flat_size
            self.dists, self.total_dist = self.calc_distances(member_states)

            # print(f"{self.cid} recalcing: \n\t{'\n\t'.join([f'{i}: {d}' for i, d in self.dists.items()])}")

        # inv_total = 1.0/max(self.dists.values())#1.0 / self.total_dist if self.total_dist > 0 else 0.0
        inv_total = 1.0 / (self.total_dist + 1e-12)

        for cid, m in self.members.items():
            # Find distance for this member
            # Find matching dist
            dist_w = self.dists[m.cid] * inv_total
            # assert dist_w <= 1, f"dist/max_dist is NOT LESSER THAN ONE, {self.dists[m.cid]}/{max(self.dists.values())} == {dist_w}"
            assert dist_w <= 1, f"dist/total is NOT LESSER THAN ONE, {self.dists[m.cid]}/{inv_total} == {dist_w} || {self.total_dist} || {self.dists}"
            one_minus_dist = (1 - dist_w)

            # For a client: update its model_state using convex combination
            if isinstance(m, client.Client):
                m.model_state = m.get_model_state()
                m.model_state_flattened = m.get_model_state_flattened()
                for key in self.get_model_state():
                    # m.model_state[key] = (self.model_state[key] * (1 - dist_w)) + (m.model_state[key] * dist_w)
                    m.model_state[key].mul_(dist_w).add_(self.get_model_state()[key], alpha=one_minus_dist)

                m.model_state_flattened = self.project_func(
                    stats.flatten(m.model_state),
                    m.model_state_flattened,
                    False,
                    True,
                    False
                )

                bigo_t += (bigo_W_size +                     # for loop
                           bigo_W_size*bigo_W_flat_size)     # flattening

                bigo_s += bigo_W_size + bigo_W_flat_size

                if not m.in_memory_swapped:
                    m.save_model_state()
                    m.save_model_state_flattened()
                    m.model_state = None
                    m.model_state_flattened = None
            else:
                # For a child cluster: update its center and then propagate further down
                # Use same formula: child center = (parent_center*(1-dist_w)) + (child_center*dist_w)
                self_model_state = self.get_model_state()
                m.model_state = m.get_model_state()
                m.model_state_flattened = m.get_model_state_flattened()

                for key in self_model_state:
                    # m.model_state[key] = (self.model_state[key] * (1 - dist_w)) + (m.model_state[key] * dist_w)
                    m.model_state[key].mul_(dist_w).add_(self_model_state[key], alpha=one_minus_dist)

                m.model_state_flattened = self.project_func(
                    stats.flatten(m.model_state), # pyright: ignore[reportArgumentType]
                    m.model_state_flattened,
                    False,
                    True,
                    False
                )

                bigo_t += (bigo_W_size +                     # for loop
                            bigo_W_size*bigo_W_flat_size)     # flattening
                
                bigo_s += bigo_W_size + bigo_W_flat_size

                if not m.in_memory_swapped:
                    m.save_model_state()
                    m.save_model_state_flattened()
                    m.model_state = None
                    m.model_state_flattened = None

                # Also update child's distances? Not needed; child will recompute when propagate_downward called on it.
                bigo_s_, bigo_t_ = m.propagate_downward(recalc_dists, bigo_W_size, bigo_W_flat_size)
                bigo_s += bigo_s_
                bigo_t += bigo_t_

        return bigo_s, bigo_t

    def get_2dpos(self, dic: dict[int, tuple[None|str, bool, tuple[float, float]]]):
        dic[self.cid] = (None, True, tuple(
                                    self.project_func(
                                        stats.flatten(self.get_model_state()), 
                                        None, 
                                        False, 
                                        False, 
                                        True
                                    ).tolist()
                                ))
        if all(isinstance(m, client.Client) for m in self.members.values()):
            # Leaf cluster
            for c in self.members.values():
                dic[c.cid] = (
                    'r' if c.client_type != ClientTypes.NORMAL else None, # pyright: ignore[reportAttributeAccessIssue]
                    False, 
                    tuple(
                        self.project_func(
                            stats.flatten(c.get_model_state()), 
                            None, 
                            False, 
                            False, 
                            True
                        ).tolist()
                    )
                )
        else:
            for c in self.members.values():
                dic[c.cid] = (
                    None,
                    True, 
                    tuple(
                        self.project_func(
                            stats.flatten(c.get_model_state()), 
                            None, 
                            False, 
                            False, 
                            True
                        ).tolist()
                    )
                )
                c.get_2dpos(dic)

    def print_tree(self, level=0, save_info: bool=False):
        indent = "  " * level
        if all(isinstance(m, client.Client) for m in self.members.values()):
            # Leaf cluster
            cids = [f"{c.cid} | {c.client_type}" for c in self.members.values()] # pyright: ignore[reportAttributeAccessIssue]
            log.info(f"{indent}Cluster in {'memory' if self.model_state is not None else 'disk'} (leaf) with clients: {', '.join(cids)}", extra={"save": save_info})
            # log.info(f"{indent}Outliers: {self.find_outlier_clients()}")
        else:
            log.info(f"{indent}Cluster in {'memory' if self.model_state is not None else 'disk'} (internal) with {len(self.members)} children", extra={"save": save_info})
            for m in self.members.values():
                if isinstance(m, Cluster):
                    m.print_tree(level+1, save_info=save_info)
                else:
                    log.info(f"{indent}  Client {m.cid} | {m.client_type} in {'memory' if m.model_state is not None else 'disk'}", extra={"save": save_info})

    @staticmethod
    def attract_aggregate(
        client_weights: dict[int, tuple[models.StateDict, torch.Tensor]],  # (state, ident, distance)
        dists: dict[int, float],
        cluster_center: models.StateDict,
        total_client_dist: float,
        self_cid: int
    ) -> models.StateDict:
        """Weighted aggregation where weight = 1 - (dist/total_dist)."""
        if not client_weights:
            return copy.deepcopy(cluster_center)

        global_weights = copy.deepcopy(cluster_center)
        weights = {}

        for key in global_weights.keys():
            total = 0.0
            assign = True

            for ident, state__flat in client_weights.items():
                weights[ident] = (total_client_dist / (dists[ident] + 1e-12)) #1 - (dists[ident] / total_client_dist) if total_client_dist > 0 else 1.0
                if assign:
                    global_weights[key] = state__flat[0][key] * weights[ident]
                    assign = False
                else:
                    global_weights[key] += state__flat[0][key] * weights[ident]

                total += weights[ident]
                # print(f"{total_client_dist=}, {dists[ident]=}, div={w}")

            global_weights[key] /= (total + 1e-12)
            # print(total)

        print(self_cid)
        for i, w in weights.items():
            print(f"\t{i}: dist: {dists[i]}, weight: {w}")

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

        T = np.quantile(list(dists.values()), q=0.5) # temperature
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

    def split(self, bigo_W_size: int, bigo_W_flat_size: int) -> int:
        bigo_t = 0
        if self.has_split:
            for m in self.members.values():
                if isinstance(m, Cluster):
                    bigo_t += m.split(bigo_W_size, bigo_W_flat_size)

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

        # if config.USE_VELOCITY:
        #     X = np.array([m.get_model_state_flattened()[:5].numpy().astype(np.float64) for m in members])
        # else:
        #     X = np.array([m.get_model_state_flattened().numpy().astype(np.float64) for m in members])

        X = np.array([m.get_model_state_flattened().numpy().astype(np.float64) for m in members])
    
        # Cluster.hdbscan_features(x, ...) returns x if use Velocity is disabled in config

        label_members: dict[int, list[int]] = {}

        for min_amt in range(min(len(self.members), 5), 1, -1):
            # dists = pairwise_distances(X).astype(np.float64)

            hdb = HDBSCAN(min_cluster_size=min_amt, copy=True, n_jobs=-1) # pyright: ignore[reportArgumentType]
            # if config.USE_VELOCITY:
            #     bigo_t += int(len(X)**2 * 5)
            # else:
            #     bigo_t += int(len(X)**2 * bigo_W_flat_size)

            bigo_t += int(len(X)**2 * bigo_W_flat_size)
            clusterer = hdb.fit(X)

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
                    bigo_t += len(X[noise_idx])*len(centers)*bigo_W_flat_size

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
        new_clusters = [
            self.new_subsplit(
                self.dists,
                i
            )
            # Cluster(i,
            #         self.dist_func, 
            #         self.project_func, 
            #         self.spill_folder, 
            #         self.in_memory_swapped,
            #         parent=self
            # ) 
            for i in new_clusters_members_dict
        ]
        bigo_t += len(new_clusters_members_dict)*bigo_W_size*bigo_W_flat_size

        if not self.in_memory_swapped:
            for c in new_clusters:
                c.swap_to_disk_checkpoint_recursive()

        self.members = {c.cid: c for c in new_clusters}

        states = {cid: (c.get_model_state(), c.get_model_state_flattened()) for cid, c in self.members.items()}
        bigo_t += len(states)*bigo_W_flat_size
        self.dists, self.total_dist = self.calc_distances(states)

        self.has_split = True

        return bigo_t

    def swap_to_disk_checkpoint_recursive(self) -> bool:
        self.save_model_state()
        self.save_model_state_flattened()

        ret = self.model_state is not None

        self.model_state = None
        self.model_state_flattened = None

        self.in_memory_swapped = False

        for m in self.members.values():
            if isinstance(m, Cluster):
                ret = ret or m.swap_to_disk_checkpoint_recursive()

        return ret

    def swap_to_memory_checkpoint_recursive(self):
        self.model_state = self.get_model_state()
        self.model_state_flattened = self.save_model_state_flattened()

        self.in_memory_swapped = True

        for m in self.members.values():
            if isinstance(m, Cluster):
                m.swap_to_memory_checkpoint_recursive()

    def clean_checkpoints(self):
        if os.path.exists(f"{self.spill_folder}/{self.cid}.chkp"):
            os.remove(f"{self.spill_folder}/{self.cid}.chkp")
        if os.path.exists(f"{self.spill_folder}/f{self.cid}.chkp"):
            os.remove(f"{self.spill_folder}/f{self.cid}.chkp")
            
        for m in self.members.values():
            m.clean_checkpoints()

    def save_metadata(self, folder: str, file_name: str):
        tree = self._build_cluster_metadata_tree()
        with open(f"{folder}/{file_name}.top", "wb") as f:
            torch.save(tree, f)

    def _build_cluster_metadata_tree(self) -> dict[str, int|str|dict]:
        tree = {
            'cid': self.cid, 
            'type': "Cluster", 
            'model_state': self.get_model_state(), 
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
