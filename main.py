import logging as log
import pickle
import types as typ
from enum import Enum
import types

import os
import ray

import lib
import config
from client_types import ClientTypes
import data_pathalogical as datap
import data2
import models
from client import Client
from server import Server, TrainProtocol

# from dataset_analysis import analyze_feature_similarity
from collections import Counter

class DataDistribution(Enum):
    Pathological = 1
    Dirchlet = 2

def default(log_file: str|None = None, data_distribution_module: types.ModuleType = datap):
    if log_file is not None: os.makedirs(config.LOG_DIR, exist_ok=True)
    lib.set_all_seeds(config.RANDOM_SEED)
    lib.set_log_level(log.INFO, log_file)

    # train_loaders, test_loaders = data.generate_federated_dataloaders(config.NUM_CLIENTS, config.DIRICHLET_ALPHA, config.BATCH_SIZE, train_test_split_ratio=0.8)
    client_splits, combined_data, full_features = data_distribution_module.generate_client_splits(
        config.NUM_CLIENTS, 
        config.DIRICHLET_ALPHA, 
        train_test_split_ratio=0.8,
        test_sampling_mode='gaussian',
        gaussian_sigma=2.0,
        n_feature_clusters=10
    )

    return client_splits, combined_data, full_features


def main(train_protocol: TrainProtocol, run_name: str, test_run_calc: bool=False, data_distribution: DataDistribution = DataDistribution.Pathological):

    module__ = datap
    if data_distribution == DataDistribution.Pathological:
        module__ = datap
    elif data_distribution == DataDistribution.Dirchlet:
        module__ = data2

    if not test_run_calc: ray.init(num_cpus=config.NUM_CPUS, num_gpus=config.NUM_GPUS)
    run_id, run_log_dir = lib.generate_new_log_run(run_name, (not test_run_calc))

    client_splits, combined_data, full_features = default(run_log_dir if not test_run_calc else None, module__)

    # if full_features is not None: 
    #     client_train_indices = [train_idx for train_idx, _ in client_splits]
    #     analyze_feature_similarity(full_features, client_train_indices)

    types = Client.sample_types(config.NUM_CLIENTS, config.CLIENT_TYPES)

    log.info(f"Sampled clients: {Counter(types)}")

    model = models.get_model()
    log.debug("Model loaded.")

    server = Server(model, types, client_splits, config.NUM_CLIENTS, config.RANDOM_SEED)

    log.info("Starting Training", extra={"save": True})
    save_folder = server.train(train_protocol, combined_data, run_id, run_log_dir, save=True, test_run_calc=test_run_calc)

    if config.WRITE_LOGS and not test_run_calc:
        with open(f"{save_folder}/metadata.npy", "wb") as f:
            config_file = {key: val for key, val in vars(config).items() if not key.startswith("__") and not isinstance(val, typ.ModuleType)}
            pickle.dump(config_file, f)

def test_backdoor(folder_name):
    import torch
    import client    
    import stats
    from functools import reduce
    import numpy as np
    import data

    client_splits, combined_data, full_features = default()

    folder_to_test = folder_name

    models_str = [f"logs/{folder_to_test}/{i}" for i in os.listdir(f"logs/{folder_to_test}") if i.endswith(".pth")]

    client_saves = []
    for m in models_str:
        with open(m, "rb") as f:
            client_saves.append(torch.load(f))

    clients = [
        client.Client(i, client_splits, lambda x:x, client_saves[i], ClientTypes.BACKDOOR_0_ALL, save_log=False, seed=config.RANDOM_SEED) # pyright: ignore[reportArgumentType]
        for i in range(len(client_saves))
    ]

    for c in clients:
        if c.attack is not None:
            c.attack.prepare(client_saves[0], models.get_model, combined_data, c.split)  # pyright: ignore[reportArgumentType]

    dataset_ref = ray.put(combined_data)
    batch_size = ray.put(config.BATCH_SIZE)
    device = ray.put(config.DEVICE)
    accs = ray.get([c.evaluate(
        dataset_ref, 
        batch_size, 
        device, 
        should_attack=True, 
        # should_attack=False, 
        calc_counts=True
        ) for c in clients])
    del dataset_ref
    del batch_size
    del device

    accs.sort(key=lambda x: x[1])
    accs_, cids, counters = zip(*accs)

    accs_lis = list(accs_)

    if counters[0] is not None:
        true_counter = Counter()
        for c_split in client_splits:
            for _, y in data.build_client_loaders(combined_data, c_split, 64, False):
                true_counter += Counter(y.cpu().numpy().tolist())
        final_counter = reduce(lambda x, y: x+y, counters)
        log.info(f"Pred: {final_counter}\n")
        log.info(f"True: {true_counter}\n")

    total_clients = len(accs_lis)
    backdoor_clients = []
    folder = config.LOG_DIR + "/" + folder_name
    file = "topology.top"

    acc = stats.avg(accs_lis)
    log.info(f"\tAccuracy: {acc*100:.2f}% | {len(accs_lis)} total clients evaluated.")

    if file in os.listdir(f"logs/{folder_to_test}"):
        with open(f"{folder}/{file}", "rb") as f:
            topology = torch.load(f)

        def print_tree(tree):
            nonlocal backdoor_clients
            if tree["type"] == "Client":
                return
            
            members = tree.get("members", {})
            if all(member["type"] == "Client" for member in members.values()):
                for client in members.values():
                    if client["attack"] == "backdoor_0":
                        backdoor_clients.append(client["cid"])
            else:
                for i, member in enumerate(members.values()):
                    print_tree(member)

        print_tree(topology)

        acc = stats.avg(accs_lis)
        no_attack_acc = []
        backdoor_attack_acc = []
        for i, acc in enumerate(accs_lis):
            if i in backdoor_clients:
                backdoor_attack_acc.append(acc)
            else:
                no_attack_acc.append(acc)

        log.info(f"\t STD: {np.array(accs_lis).std()}")

        log.info(f"\tATTACK ACCURACY: {np.average(backdoor_attack_acc)}")
        log.info(f"\tNO ATTACK ACCURACY: {np.average(no_attack_acc)}")
    else:    
        log.info(f"\tAll accs:")
        for i, acc in enumerate(accs_lis):
            log.info(f"{i}: {acc}")

        log.info(f"\t STD: {np.array(accs_lis).std()}")

def blind_test_backdoor(folder_name):
    import torch
    import test_backdoor as tb
    import numpy as np

    lib.set_all_seeds(config.RANDOM_SEED)
    lib.set_log_level(log.INFO)

    folder_to_test = folder_name
    models_str = [f"logs/{folder_to_test}/{i}" for i in os.listdir(f"logs/{folder_to_test}") if i.endswith(".pth")]
    models_str = sorted(models_str, key=lambda x: int((x.split("/")[-1]).split("_")[0]))

    client_saves = []
    for m in models_str:
        with open(m, "rb") as f:
            client_saves.append(torch.load(f))

    anomalies = []
    idxs = []
    spec_scores = []


    for i, c in enumerate(client_saves):
        max_anomaly, max_idx = tb.dfbscanner_detect(c, 10)
        anomalies.append(max_anomaly)
        idxs.append(max_idx)

        spec_scores.append(tb.spectral_signature_detect(c))

        log.info(f"{i}: {max_anomaly} score | {max_idx} idx")

    log.info(f"\tAvg DFBS Anomaly Score: {np.mean(anomalies)} | STD: {np.std(anomalies)}")
    log.info(f"\tAvg DFBS Anomaly IDX: {np.mean(idxs)} | STD: {np.std(idxs)}")

    log.info(f"\tAvg Spectral Anomaly Score: {np.mean(spec_scores)} | STD: {np.std(spec_scores)}")

def blind_test_backdoor_topology(folder_name):
    import torch
    import test_backdoor as tb
    import numpy as np

    lib.set_all_seeds(config.RANDOM_SEED)
    lib.set_log_level(log.INFO)

    folder = config.LOG_DIR + "/" + folder_name
    file = "topology.top"
    with open(f"{folder}/{file}", "rb") as f:
        topology = torch.load(f)

    def print_tree(tree, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        # indent = "  " * level

        if tree["type"] == "Client":
            # print(f"{indent}Client {tree['cid']}")
            return
        
        members = tree.get("members", {})
        log.info(f"{prefix}{connector}Cluster #{tree['cid']} with {len(members)} children")
        
        model_state = tree["model_state"]
        max_anomaly, max_idx = tb.dfbscanner_detect(model_state, 10)
        log.info(f"{prefix}{connector}  Score: {max_anomaly} | idx: {max_idx}")

        # Check if this is a leaf cluster
        if all(member["type"] == "Client" for member in members.values()):
            cids = [str(member["cid"]) for member in members.values()]
            log.info(f"{prefix}{connector} -- Clients: {', '.join(cids)}")

            anomalies = []
            idxs = []

            for client in members.values():
                with open(f"{folder}/{client["model_state_file"]}", "rb") as f:
                    model_state = torch.load(f)

                max_anomaly, max_idx = tb.dfbscanner_detect(model_state, 10)
                anomalies.append(max_anomaly)
                idxs.append(max_idx)

                log.info(f"{prefix}{connector} --   Client #{client["cid"]}:")
                log.info(f"{prefix}{connector} --     Score: {max_anomaly} | idx: {max_idx}")
                log.info(f"{prefix}{connector} --     Attacked: {client["attack"]}")

            log.info(f"{prefix}{connector} -- Avg Score: {np.average(anomalies)} | STD: {np.std(anomalies)}")

        else:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, member in enumerate(members.values()):
                print_tree(member, prefix=child_prefix, is_last=(i == len(members) - 1))

    print_tree(topology)    

def PRINT_TREE(folder_name):
    import torch
    lib.set_all_seeds(config.RANDOM_SEED)
    lib.set_log_level(log.INFO)

    file = "topology.top"
    with open(f"logs/{folder_name}/{file}", "rb") as f:
        topology = torch.load(f)

    def print_tree(tree, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        # indent = "  " * level

        if tree["type"] == "Client":
            # print(f"{indent}Client {tree['cid']}")
            return
        
        members = tree.get("members", {})
        log.info(f"{prefix}{connector}Cluster #{tree['cid']} with {len(members)} children")

        model_state = tree["model_state"] # model state of the cluster

        # Check if this is a leaf cluster
        if all(member["type"] == "Client" for member in members.values()):
            cids = [str(member["cid"]) for member in members.values()]
            log.info(f"{prefix}{connector} -- Clients: {', '.join(cids)}")

            for client in members.values():
                with open(f"logs/{folder_name}/{client["model_state_file"]}", "rb") as f:
                    model_state = torch.load(f) # model state of the client
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, member in enumerate(members.values()):
                print_tree(member, prefix=child_prefix, is_last=(i == len(members) - 1))

    print_tree(topology)

def evaluate_entropy(folder_name):
    import torch

    lib.set_all_seeds(config.RANDOM_SEED)
    lib.set_log_level(log.INFO)

    folder = config.LOG_DIR + "/" + folder_name
    file = "topology.top"
    with open(f"{folder}/{file}", "rb") as f:
        topology = torch.load(f)

    def evaluate_tree(root):
        # -------------------------------
        # Pass 1: Count labels
        # -------------------------------
        def count(node):
            if node["type"] == "Client":
                if node["attack"] == "backdoor_0":
                    node["n0"] = 1
                    node["n1"] = 0
                else:
                    node["n0"] = 0
                    node["n1"] = 1
                return node["n0"], node["n1"]

            n0 = 0
            n1 = 0
            for child in node["members"].values():
                c0, c1 = count(child)
                n0 += c0
                n1 += c1

            node["n0"] = n0
            node["n1"] = n1
            return n0, n1

        count(root)

        total_score = 0.0
        total_leaves = 0

        backdoor_score = 0.0
        backdoor_count = 0

        benign_score = 0.0
        benign_count = 0

        # -------------------------------
        # Pass 2: Average purity along path
        # -------------------------------
        def traverse(node, purity_sum, depth):
            nonlocal total_score
            nonlocal total_leaves
            nonlocal backdoor_score
            nonlocal backdoor_count
            nonlocal benign_score
            nonlocal benign_count

            total = node["n0"] + node["n1"]
            purity = max(node["n0"], node["n1"]) / total

            purity_sum += purity
            depth += 1

            if node["type"] == "Client":
                avg_path_purity = purity_sum / depth
                node["path_purity"] = avg_path_purity

                total_score += avg_path_purity
                total_leaves += 1

                if node["attack"] == "backdoor_0":
                    backdoor_score += avg_path_purity
                    backdoor_count += 1
                else:
                    benign_score += avg_path_purity
                    benign_count += 1

                return

            for child in node["members"].values():
                traverse(child, purity_sum, depth)

        traverse(root, 0.0, 0)

        return {
            "average_path_purity": total_score / total_leaves,
            "average_backdoor_path_purity": (
                backdoor_score / backdoor_count if backdoor_count else None
            ),
            "average_benign_path_purity": (
                benign_score / benign_count if benign_count else None
            ),
        }

    score = evaluate_tree(topology)

    print(score)

if __name__ == "__main__":
    # folder_to_test = "RUN_Tue_Aug__4_14-21-24_2026_backdoor_fedattract_pathalogical_diff_test_scaled"
    # folder_to_test = "RUN_Fri_Jul_31_22-02-39_2026_backdoor_fedattract_pathalogical"
    # folder_to_test = "RUN_Tue_Aug__4_12-09-36_2026_backdoor_fedattract_pathalogical_diff_test"

    # folder_to_test = "RUN_Sat_Jul_25_16-19-27_2026_backdoor_pathalogical_cap"
    # folder_to_test = "RUN_Sat_Jul_25_18-36-25_2026_backdoor_avg"

    main(
        TrainProtocol.FedCap,
        "alie30_fedcap_pathalogical", 
        test_run_calc=False,
        data_distribution=DataDistribution.Pathological
    )
    # test_backdoor(folder_to_test)
    # blind_test_backdoor(folder_to_test)
    # blind_test_backdoor_topology(foslder_to_test)
    # evaluate_entropy(folder_to_test)
