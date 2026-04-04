import torch
import logging as log

import models
import client
import stats

from typing import Callable, Sequence
# from itertools import chain
import copy

class Cluster:
    def __init__(
            self, 
            clients: list[client.Client], 
            dist_func: Callable[[models.StateDict, torch.Tensor], float],
            cluster_center: models.StateDict|None = None,
        ) -> None:
        
        self.clients = clients
        self.clients.sort(key=lambda c: c.cid)

        # self.client_weights = [c.model_state for c in self.clients]
        self.client_cid_idx = {c.cid: i for i, c in enumerate(self.clients)} # as a cluster only holds a subsection of actual client, map client.cid -> self.clients[idx]

        self.dist_func = dist_func

        client_weights = [(c.model_state, c.cid) for c in self.clients]

        self.cluster_center: models.StateDict = cluster_center or Cluster.avg_model_states(client_weights)
        self.cluster_flattened = stats.flatten(self.cluster_center)

        self.dists, self.total_dist = self.calc_distances(client_weights)
        log.info(f"1: {self.dists}")

    def move_cluster_center(self, client_weights: list[client.ModelState]):
        self.cluster_center = Cluster.attract_aggregate(
            client_weights, 
            self.cluster_center, 
            self.dists, 
            self.total_dist
        )
        self.cluster_flattened = stats.flatten(self.cluster_center)

    def calc_distances(self, state_list: Sequence[tuple[models.StateDict, int]]):
        dists: list[tuple[float, int]] = []
        total_dist = 0

        for state in state_list:
            d = self.dist_func(
                    state[0], 
                    self.cluster_flattened
                )
            dists.append((d, state[1]))
            total_dist += d

        #assert dist[i][1] in ascending order
        prev = dists[0][1]
        for i in range(1, len(dists)):
            if dists[i][1] < prev:
                raise ValueError("Client list HAS to be sorted")

        return dists, total_dist
    
    def calc_distances_inplace(self, weights__cid: Sequence[tuple[client.ModelState, int]]):
        client_selected_weights: list[client.ModelState] = []
        selected_cid: list[int] = []

        if len(weights__cid) != 0:
            client_selected_weights, selected_cid = zip(*weights__cid) # pyright: ignore[reportAssignmentType]

        not_selected_cids = list(set(self.client_cid_idx.keys()) - set(selected_cid)) # not selected from this cluster

        client_weights = [(w__c[0][0], w__c[1]) for w__c in weights__cid] + [(c.model_state, c.cid) for c in self.clients if c.cid in not_selected_cids]
        client_weights.sort(key=lambda x: x[1])

        self.dists, self.total_dist = self.calc_distances(client_weights)
        # self.dists.sort(key=lambda x: x[1])

        # log.info(f"2: {self.dists} | {[i[1] for i in client_weights]}")

        assert len(client_weights) == len(self.dists) == len(self.clients)

        return client_weights

    def move_new_client_weights(self, local_sds__cid: Sequence[tuple[client.ModelState, int]]):
        updated_clients = {cid: local_sd for local_sd, cid in local_sds__cid}

        for client in self.clients:
            assert client.cid == self.dists[client.cid][1], f"{client.cid} != {self.dists[client.cid]}"

            dist_w = self.dists[client.cid][0]/self.total_dist

            if client.cid in updated_clients:
                client.model_state = updated_clients[client.cid][0]

            for key in self.cluster_center: # for each param in clients
                client.model_state[key] = (self.cluster_center[key] * (1-dist_w)) + (client.model_state[key] * dist_w)
                # more further away from cluster center => more weightage to local param

    # FedAttractAVG
    @staticmethod
    def attract_aggregate(
        client_weights: Sequence[client.ModelState], 
        cluster_center: models.StateDict, 
        client_dists: list[tuple[float, int]], 
        total_client_dist: float
    ) -> models.StateDict:
        # total all of weights
        
        assert len(client_weights) == len(client_dists)

        # global_weights = client_weights[0][0] #copy.deepcopy(cluster_center)
        global_weights = copy.deepcopy(cluster_center)

        for key in global_weights.keys(): # for each param in clients
            global_weights[key] = (1-(client_dists[0][0] / total_client_dist)) * client_weights[0][0][key]

            for i in range(1, len(client_weights)): # for each client
                global_weights[key] += client_weights[i][0][key] * (1-(client_dists[i][0] / total_client_dist))

            # avg
            total_weight = sum(1 - di/total_client_dist for di, _ in client_dists)
            global_weights[key] /= total_weight
            # global_weights[key] /= len(client_weights)

        return global_weights

    @staticmethod
    def avg_model_states(client_weights: Sequence[client.ModelState]) -> models.StateDict:
        global_weights = copy.deepcopy(client_weights[0][0])

        for key in global_weights.keys(): # for each param in clients
            global_weights[key] = client_weights[0][0][key].clone()
                
            for i in range(1, len(client_weights)): # for each client
                global_weights[key] += client_weights[i][0][key]

            # avg
            global_weights[key] /= len(client_weights)
                
        return global_weights
    
