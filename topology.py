# topology.py
import torch
import logging as log
import copy
import numpy as np
from typing import Callable, Sequence
import models
import client
import stats
# from sklearn.cluster import DBSCAN
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4
import config

from collections import OrderedDict

import threading

class Cluster:
    def __init__(
            self,
            members: dict[int, 'client.Client|Cluster'],
            dist_func: Callable[[torch.Tensor, torch.Tensor], float],
            project_func: Callable[[torch.Tensor], torch.Tensor],
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
        self.model_state_flattened = self.project_func(stats.flatten(self.model_state))

        # distances and total_dist will be recomputed during updates
        self.dists, self.total_dist = self.calc_distances({cid: (c.model_state, c.model_state_flattened) for cid, c in self.members.items()}) #dict[int, tuple[models.StateDict, torch.Tensor]]
        self.has_split = False


        # X = np.array([m.model_state_flattened.numpy() for m in self.members.values()])
        # self.pairwise = pairwise_distances(X)

        # self.ema_alpha = 0.9
        # self.ema_mean = self.pairwise.mean()
        # self.ema_var = 0.0

        # log.info(f"\t\t\tINIT Mean & var: {self.ema_mean}, {self.ema_var}")

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
            d = self.dist_func(state__flat[1], self.model_state_flattened)
            dists[ident] = d
            total_dist += d
        return dists, total_dist

    def update_centers_upward(self, updated_client_states: dict[int, models.StateDict]) -> None:
        """
        Recursively update cluster centers from leaves upward.
        updated_client_states: dict mapping client.cid -> new state_dict (after local training).
        """
        # First, update child clusters (if any) and collect their current states
        member_states: dict[int, tuple[models.StateDict, torch.Tensor]] = {}
        for cid, m in self.members.items():
            if isinstance(m, client.Client):                
                member_states[m.cid] = (m.model_state, m.model_state_flattened)
            else:
                # Recursively update child cluster
                m.update_centers_upward(updated_client_states)
                # After child cluster updated, its center is new
                member_states[m.cid] = m.model_state, m.model_state_flattened

        # Now compute distances for this cluster
        self.dists, self.total_dist = self.calc_distances(member_states)

        # X = np.array([m.model_state_flattened.numpy() for m in self.members.values()])
        # self.pairwise = pairwise_distances(X)

        # pair_mean = self.pairwise.mean()
        # pair_var = self.pairwise.var()

        # diff = pair_mean - self.ema_mean
        # self.ema_mean += self.ema_alpha * diff
        # self.ema_var = (
        #     (1 - self.ema_alpha) * self.ema_var
        #     + self.ema_alpha * pair_var
        #     + self.ema_alpha * (1 - self.ema_alpha) * diff * diff
        # )
        # log.info(f"\t\t\tChanged EMA mean and Var: {self.ema_mean} {self.ema_var}")

        # Update this cluster's center using attract_aggregate
        # Build list of (state, dist, ident) for attract_aggregate
        # children_weights: list[tuple[models.StateDict, float, int]] = []
        # for (state, ident), (dist, ident2) in zip(member_states, self.dists):
        #     assert ident == ident2, "Dist calc, has shuffled in update_clusters_upward"
        #     children_weights.append((state, dist, ident))

        self.model_state = Cluster.attract_aggregate(
            member_states, self.dists, self.model_state, self.total_dist
        )
        self.model_state_flattened = self.project_func(stats.flatten(self.model_state))

    def propagate_downward(self) -> None:
        """
        From root downward, update child clusters and clients using the new center and distances.
        """
        # First update this cluster's own members (clients or child clusters)
        inv_total = 1.0 / self.total_dist if self.total_dist > 0 else 0.0
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

                m.model_state_flattened = self.project_func(stats.flatten(m.model_state))
            else:
                # For a child cluster: update its center and then propagate further down
                # Use same formula: child center = (parent_center*(1-dist_w)) + (child_center*dist_w)
                for key in self.model_state:
                    # m.model_state[key] = (self.model_state[key] * (1 - dist_w)) + (m.model_state[key] * dist_w)
                    m.model_state[key].mul_(dist_w).add_(self.model_state[key], alpha=one_minus_dist)

                m.model_state_flattened = self.project_func(stats.flatten(m.model_state))
                # Also update child's distances? Not needed; child will recompute when propagate_downward called on it.
                m.propagate_downward()

    # def propagate_downward_inv(self) -> None:
    #     """
    #     From root downward, update child clusters and clients using the new center and distances.
    #     """
    #     # First update this cluster's own members (clients or child clusters)
    #     # inv_total = 1.0 / self.total_dist if self.total_dist > 0 else 0.0
    #     for cid, m in self.members.items():
    #         # Find distance for this member
    #         # Find matching dist
    #         dist_w = self.total_dist / self.dists[m.cid]

    #         # For a client: update its model_state using convex combination
    #         if isinstance(m, client.Client):
    #             for key in self.model_state:
    #                 # m.model_state[key] = (self.model_state[key] * (1 - dist_w)) + (m.model_state[key] * dist_w)
    #                 m.model_state[key].mul_(dist_w).add_(self.model_state[key], alpha=one_minus_dist)

    #             m.model_state_flattened = self.project_func(stats.flatten(m.model_state))
    #         else:
    #             # For a child cluster: update its center and then propagate further down
    #             # Use same formula: child center = (parent_center*(1-dist_w)) + (child_center*dist_w)
    #             for key in self.model_state:
    #                 # m.model_state[key] = (self.model_state[key] * (1 - dist_w)) + (m.model_state[key] * dist_w)
    #                 m.model_state[key].mul_(dist_w).add_(self.model_state[key], alpha=one_minus_dist)

    #             m.model_state_flattened = self.project_func(stats.flatten(m.model_state))
    #             # Also update child's distances? Not needed; child will recompute when propagate_downward called on it.
    #             m.propagate_downward()

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

    def split(self):
        # hardcoding for now, based on set seed
        if self.has_split:
            for m in self.members.values():
                if isinstance(m, Cluster):
                    m.split()

            return
        
        if len(self.members) == 2:
            log.info(f"Cluster '{self.cid}' has reached min_cluster_size")
            return

        members: list['client.Client'] = []
        member_cid: list[int] = []

        for cid, mem in self.members.items():
            assert isinstance(mem, client.Client), f"{mem.cid} in cluster {self.cid} is supposed to be a CLIENT, not a CLuster."
            member_cid.append(cid)
            members.append(mem)

        X = np.array([m.model_state_flattened.numpy() for m in members])
        dists = pairwise_distances(X)

        if not np.allclose(dists, dists.T, atol=1e-12):
            log.info(f"\t\t!!! DISTS MATRIX WAS ASYMMETRIC {dists}\n\n")
            dists = (dists + dists.T) / 2

        label_members: dict[int, list[int]] = {}

        for min_amt in range(min(len(self.members), 5), 1, -1):
            hdb = HDBSCAN(min_cluster_size=min_amt, copy=True, metric='precomputed', n_jobs=-1) # pyright: ignore[reportArgumentType]
            clusterer = hdb.fit(dists)

            labels_all: np.ndarray = clusterer.labels_.copy()

            log.info(f"\t\t{labels_all=}")
            # log.info(f"\t\tDist stas: {dists.mean()=}, {dists.std()=}")

            unique_clusters = set(labels_all) - {-1}

            # Find noise points
            noise_idx = np.where(labels_all == -1)[0]

            # Assign each noise point to nearest cluster center
            if len(noise_idx) != 0 and len(unique_clusters) != 0:
                centers = np.array([X[labels_all == c].mean(axis=0) for c in unique_clusters])
                
                nearest, _ = pairwise_distances_argmin_min(X[noise_idx], centers)
                unique_clusters_list = list(unique_clusters)

                for i, idx in enumerate(noise_idx):
                    labels_all[idx] = unique_clusters_list[nearest[i]]

            label_members.clear()
            assert len(member_cid) == len(labels_all)
            for cid, l in zip(member_cid, labels_all):
                label_members.setdefault(l.item(), []).append(cid)

            if len(label_members) != 1:
                log.info(f"Cluster '{self.cid}' splitting with min cluster size: {min_amt}")
                break

        log.info(f"\t\t{label_members=}")
        if len(label_members) == 1:
            log.info(f"Cluster '{self.cid}' children aren't diverse to form other clusters.")
            return
        
        # def _run():

        #     X2 = PCA(n_components=2).fit_transform(X)
        #     colors = {}
        #     for label in label_members.keys():
        #         if label == -1:
        #             colors[label] = 'gray'
        #         else:
        #             # Generate random color
        #             colors[label] = np.random.rand(3,)

        #     plt.figure()
        #     for label, clients in label_members.items():
        #         mask = labels_all == label
        #         color = colors[label]
        #         label_name = f'Label {label}' if label != -1 else 'Noise/Outlier'
        #         plt.scatter(X2[mask, 0], X2[mask, 1],
        #                 c=[color], label=label_name, alpha=0.7, s=50)
                
        #     plt.show()

        # thread = threading.Thread(target=_run)
        # thread.start()

        ####################################################

        new_clusters_members_dict = [{idx: self.members[idx] for idx in cluster} for label_id, cluster in label_members.items()]
        new_clusters = [Cluster(i, self.dist_func, self.project_func, self) for i in new_clusters_members_dict]

        self.members = {c.cid: c for c in new_clusters}

        self.dists, self.total_dist = self.calc_distances({cid: (c.model_state, c.model_state_flattened) for cid, c in self.members.items()}) #dict[int, tuple[models.StateDict, torch.Tensor]]

        # self.ema_alpha = 0.1
        # self.ema_mean = stats.avg(self.dists.values())
        # self.ema_var = 0.0

        # log.info(f"\t\tSPLIT Mean & var: {self.ema_mean}, {self.ema_var}")

        self.has_split = True

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

    # def find_outlier_clients(self):
    #     std = self.ema_var**0.5
    #     max_d = (self.ema_mean + 1.5*std)

    #     cids: list[int] = []

    #     for cid, d in self.dists.items():
    #         if d > max_d:
    #             cids.append(cid)

    #     return cids