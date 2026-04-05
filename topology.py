# topology.py
import torch
import logging as log
import copy
import numpy as np
from typing import Callable, OrderedDict, Sequence, Union
import models
import client
import stats
from sklearn.cluster import KMeans  # add to requirements
from uuid import uuid4
import config

class Cluster:
    def __init__(
            self,
            members: dict[int, 'client.Client|Cluster'],
            dist_func: Callable[[torch.Tensor, torch.Tensor], float],
            project_func: Callable[[torch.Tensor], torch.Tensor],
            parent: 'Cluster|None' = None,
            cluster_center: models.StateDict|None = None,
        ) -> None:
        self.members = members
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

        self.ema_alpha = 0.1
        self.ema_mean = stats.avg(self.dists.values())
        self.ema_var = 0.0

        log.info(f"\t\t\tINIT Mean & var: {self.ema_mean}, {self.ema_var}")

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
        diff = stats.avg(self.dists.values()) - self.ema_mean
        self.ema_mean += self.ema_alpha * diff
        self.ema_var = (1 - self.ema_alpha) * (self.ema_var + self.ema_alpha * diff * diff)
        log.info(f"\t\t\tChanged EMA mean and Var: {self.ema_mean} {self.ema_var}")

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
            dist_w = self.dists.get(m.cid, 0.0) * inv_total
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

    # def split(self, round_id: int, n_clusters: int = 2) -> bool:
    #     """
    #     Split this cluster if it contains only clients (leaf) and the number of clients >= n_clusters*2.
    #     Performs k-means on flattened client states and creates sub-clusters.
    #     Returns True if split occurred.
    #     """
    #     # Only split if all members are clients (no sub-clusters yet)
    #     if not all(isinstance(m, client.Client) for m in self.members):
    #         return False
    #     if len(self.members) < n_clusters * 2:
    #         return False

    #     # Collect client states as feature vectors
    #     clients = self.members
    #     states_flattened = [stats.flatten(c.model_state).numpy() for c in clients]
    #     X = np.stack(states_flattened)

    #     kmeans = KMeans(n_clusters=n_clusters, random_state=round_id, n_init=10)
    #     labels = kmeans.fit_predict(X)

    #     # Create new clusters
    #     new_clusters = []
    #     for k in range(n_clusters):
    #         cluster_clients = [clients[i] for i, lbl in enumerate(labels) if lbl == k]
    #         if len(cluster_clients) > 0:
    #             subcluster = Cluster(
    #                 members=cluster_clients,
    #                 dist_func=self.dist_func,
    #                 parent=self,
    #                 cluster_center=None
    #             )
    #             new_clusters.append(subcluster)

    #     # Replace members with sub-clusters
    #     self.members = new_clusters
    #     log.info(f"Round {round_id}: Cluster split into {len(new_clusters)} sub-clusters")
    #     return True

    def print_tree(self, level=0):
        indent = "  " * level
        if all(isinstance(m, client.Client) for m in self.members):
            # Leaf cluster
            cids = [str(c.cid) for c in self.members.values()]
            log.info(f"{indent}Cluster (leaf) with clients: {', '.join(cids)}")
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
                w = 1 - (dists[ident] / total_client_dist) if total_client_dist > 0 else 1.0
                if assign:
                    global_weights[key] = state__flat[0][key] * w
                    assign = False
                else:
                    global_weights[key] += state__flat[0][key] * w

                total += w

            global_weights[key] /= total

        return global_weights
    
    # @staticmethod
    # def attract_aggregate(
    #     client_weights: dict[int, tuple[models.StateDict, torch.Tensor]],
    #     dists: dict[int, float],
    #     cluster_center: models.StateDict,
    #     total_client_dist: float
    # ) -> models.StateDict:
    #     """Weighted aggregation where weight = 1 - (dist/total_dist), vectorized."""

    #     if not client_weights:
    #         return copy.deepcopy(cluster_center)

    #     global_weights = copy.deepcopy(cluster_center)

    #     # Precompute weights tensor
    #     idents = list(client_weights.keys())

    #     if total_client_dist > 0:
    #         weights = torch.tensor(
    #             [1 - (dists[i] / total_client_dist) for i in idents],
    #             dtype=torch.float32
    #         )
    #     else:
    #         weights = torch.ones(len(idents), dtype=torch.float32)

    #     # Normalize once (avoids repeated division per key)
    #     weights = weights / weights.sum()

    #     for key in global_weights.keys():
    #         # Stack all client tensors for this key
    #         stacked = torch.stack([
    #             client_weights[i][0][key] for i in idents
    #         ], dim=0)  # shape: (num_clients, ...)

    #         # Reshape weights for broadcasting
    #         w = weights.view(-1, *([1] * (stacked.dim() - 1)))

    #         # Weighted sum
    #         global_weights[key] = torch.sum(stacked * w, dim=0)

    #     return global_weights

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

    def find_outlier_clients(self):
        std = self.ema_var**0.5
        max_d = (self.ema_mean + 1.5*std)

        cids: list[int] = []

        for cid, d in self.dists.items():
            if d > max_d:
                cids.append(cid)

        return cids