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

    log.info(f"Sampled clients: {Counter(types)}", extra={'save': True})

    model = models.get_model()
    log.debug("Model loaded.")

    server = Server(
        model, 
        types, 
        client_splits, 
        config.NUM_CLIENTS,
        spill_to_disk=False,
        spill_folder="spill", 
        seed=config.RANDOM_SEED
    )

    log.info("Starting Training", extra={"save": True})
    try:
        save_folder = server.train(train_protocol, combined_data, run_id, run_log_dir, save=True, test_run_calc=test_run_calc)
    except KeyboardInterrupt:
        server.global_cluster.clean_checkpoints()
        raise

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
        client.Client(
            i, 
            client_splits, 
            lambda x:x, 
            client_saves[i], 
            "spill",
            ClientTypes.BACKDOOR_0_ALL, 
            save_log=False, 
            seed=config.RANDOM_SEED
        ) # pyright: ignore[reportArgumentType]
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
    accs_, cids, counters, _ = zip(*accs)

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

def test_labelswitch(folder_name):
    import torch
    import client    
    import stats
    from functools import reduce
    import numpy as np
    import data
    import json

    client_splits, combined_data, full_features = default()

    folder_to_test = folder_name

    models_str = [f"logs/{folder_to_test}/{i}" for i in os.listdir(f"logs/{folder_to_test}") if i.endswith(".pth")]

    client_saves = []
    for m in models_str:
        with open(m, "rb") as f:
            client_saves.append(torch.load(f))
    with open(f"logs/{folder_name}/label_switch.json", "r") as f:
        label_switch = json.load(f)

    types = Client.sample_types(config.NUM_CLIENTS, config.CLIENT_TYPES)

    clients = [
        client.Client(
            i, 
            client_splits, 
            lambda x:x, 
            client_saves[i], 
            "spill",
            types[i], 
            save_log=False, 
            seed=config.RANDOM_SEED
        ) # pyright: ignore[reportArgumentType]
        for i in range(len(client_saves))
    ]

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
        ) for c in clients if c.client_type != ClientTypes.LABEL_SWITCH])
    del dataset_ref
    del batch_size
    del device

    accs.sort(key=lambda x: x[1])
    accs_, cids, zipp, _ = zip(*accs)
    accs_lis = list(accs_)

    attack_rate = [0, 0] # (num of instances where real == s, but pred == t)/(num instances of real == s, in our case = 100%, as we simulate ALL classes being swapped)
    ## labelswitch s->t

    # print(zipp)
    for c, zipp_res in enumerate(zipp):
        for pred, real in zipp_res:
            attack_rate[1] += 1

            if pred == int(label_switch[str(real)]):
                attack_rate[0] += 1

    acc = stats.avg(accs_lis)
    log.info(f"\tAccuracy: {acc*100:.2f}% | {len(accs_lis)} total clients evaluated.")
    log.info(f"\tAttack rate: {attack_rate[0]/attack_rate[1]}")

def generalize_test(run_log_dir):
    import json
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import seaborn as sns
    import pandas as pd

    file = config.LOG_DIR + "/" + run_log_dir + "/generalization.json"
    with open(file, "r") as f:
        gen_data: dict[str, dict[str, dict[str, float]]] = json.load(f)

    acc = pd.DataFrame({
        row: {col: values["acc"] for col, values in cols.items()}
        for row, cols in gen_data.items()
    }).T
    ax = sns.heatmap(acc, cmap="viridis")
    # Mark cells where main != "normal"
    for i, row in enumerate(acc.index):
        for j, col in enumerate(acc.columns):
            if gen_data[row][col]["main_client_type"] != "normal":
                ax.add_patch(
                    Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor="red",
                        linewidth=3
                    )
                )
                ax.text(
                    j + 0.5, i + 0.5,
                    "X",
                    color="red",
                    fontsize=20,
                    ha="center",
                    va="center"
                )

    clients = len(acc)

    plt.xlabel("Datasets")
    plt.ylabel("Models")
    plt.show()

    verify_acc = [data["acc"] for i, J in gen_data.items() for j, data in J.items() if i == j and data["main_client_type"] == "normal"]
    verify_acc = sum(verify_acc)/len(verify_acc)

    personalization = sum([data["acc"] for i, J in gen_data.items() for j, data in J.items() if i == j])/clients
    cross = sum([data["acc"] for i, J in gen_data.items() for j, data in J.items() if i != j])/((clients*(clients-1)))

    print(f"Accuracy only on Normal clients: {verify_acc}\n")
    print(f"Personalized Accuracy: {personalization}")
    print(f"Generalized Accuracy: {cross}\n")

    print(f"Personalization Gain: {personalization-cross} % points")
    print(f"Cross-client accuracy retention: {cross/personalization}")

    personalization = [data["acc"] for i, J in gen_data.items() for j, data in J.items() if i == j and data["main_client_type"] == "normal"]
    cross = [data["acc"] for i, J in gen_data.items() for j, data in J.items() if i != j and data["main_client_type"] == "normal"]

    personalization = sum(personalization)/len(personalization)
    cross = sum(cross)/len(cross)

    print(f"\n\nNORMAL ONLY:\nPersonalized Accuracy: {personalization}")
    print(f"Generalized Accuracy: {cross}\n")

    print(f"Personalization Gain: {personalization-cross} % points")
    print(f"Cross-client accuracy retention: {cross/personalization}")

    personalization = [data["acc"] for i, J in gen_data.items() for j, data in J.items() if i == j and data["main_client_type"] == "normal"]
    cross = [data["acc"] for i, J in gen_data.items() for j, data in J.items() if i != j and data["main_client_type"] == "normal" and data["sec_client_type"] == "normal"]

    personalization = sum(personalization)/len(personalization)
    cross = sum(cross)/len(cross)

    print(f"\n\nBOTH NORMAL ONLY:\nPersonalized Accuracy: {personalization}")
    print(f"Generalized Accuracy: {cross}\n")

    print(f"Personalization Gain: {personalization-cross} % points")
    print(f"Cross-client accuracy retention: {cross/personalization}")
    
def run_generalize(run_log_dir, data_distribution: DataDistribution):
    import torch
    module__ = datap
    if data_distribution == DataDistribution.Pathological:
        module__ = datap
    elif data_distribution == DataDistribution.Dirchlet:
        module__ = data2

    client_splits, combined_data, full_features = default(None, module__)
    # print(client_splits[0][0][:10])
    # return
    types = Client.sample_types(config.NUM_CLIENTS, config.CLIENT_TYPES)

    log.info(f"Sampled clients: {Counter(types)}")

    model = models.get_model()

    server = Server(
        model, 
        types, 
        client_splits, 
        config.NUM_CLIENTS,
        spill_to_disk=False,
        spill_folder="spill", 
        seed=config.RANDOM_SEED
    )

    folder = config.LOG_DIR + "/" + run_log_dir

    models_str = [f"{folder}/{i}" for i in os.listdir(folder) if i.endswith(".pth")]
    models_str__cid = [(i, int((i.split("/")[-1]).split("_")[0])) for i in models_str] #sorted(models_str, key=lambda x: int((x.split("/")[-1]).split("_")[0]))
    for m, cid in models_str__cid:
        with open(m, "rb") as f:
            server.clients[cid].model_state = torch.load(f)

    dataset_ref = ray.put(combined_data)
    batch_size = ray.put(config.BATCH_SIZE)
    device = ray.put(config.DEVICE)

    log.info("Starting Generalizaion Matrix")
    server.run_generalization_matrix_eval(dataset_ref, batch_size, device, folder)

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
    # gui__ = Process(
    #     target=gui.gui_process,
    #     args=(gui.queue,),
    #     daemon=True,
    # )

    # gui__.start()

    import sys
    if len(sys.argv) < 1:
        exit()

    if sys.argv[0] == "0":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 1.0,

            # ClientTypes.NORMAL: 0.4,
            # ClientTypes.MODEL_REPLACE_POS: 0.6,

            # ClientTypes.NORMAL: 0.4,
            # ClientTypes.BACKDOOR_0: 0.6,

            # ClientTypes.NORMAL: 0.4,
            # ClientTypes.LABEL_SWITCH: 0.6,

            # ClientTypes.NORMAL: 0.4,
            # ClientTypes.IPM: 0.6,

            # ClientTypes.NORMAL: 0.4,
            # ClientTypes.RANDOM: 0.6,

            # ClientTypes.NORMAL: 0.4,
            # ClientTypes.SIGN_FLIP: 0.6
        }
        main(
            TrainProtocol.FedAttract,
            "normal_fedattract_pathalogical", 
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "1":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.MODEL_REPLACE_POS: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "modelreplace60_fedattract_pathalogical", 
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "2":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.IPM: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "ipm60_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "3":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.ALIE: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "alie60_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "4":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.RANDOM: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "random60_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "5":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.SIGN_FLIP: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "signflip60_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "6":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.BACKDOOR_0: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "backdoor060_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "7":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.SUBTLE: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "subtle60_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "8":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.4,
            ClientTypes.LABEL_SWITCH: 0.6,
        }
        main(
            TrainProtocol.FedAttract,
            "labelswitch60_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )

    elif sys.argv[0] == "9":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.MODEL_REPLACE_POS: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "modelreplace80_fedattract_pathalogical", 
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "10":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.IPM: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "ipm80_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "11":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.ALIE: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "alie80_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "12":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.RANDOM: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "random80_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "13":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.SIGN_FLIP: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "signflip80_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "14":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.BACKDOOR_0: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "backdoor080_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "15":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.SUBTLE: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "subtle80_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    elif sys.argv[0] == "16":
        config.CLIENT_TYPES = {
            ClientTypes.NORMAL: 0.2,
            ClientTypes.LABEL_SWITCH: 0.8,
        }
        main(
            TrainProtocol.FedAttract,
            "labelswitch80_fedattract_pathalogical",
            test_run_calc=False,
            data_distribution=DataDistribution.Pathological
        )
    # run_generalize("RUN_Tue_Oct__6_16-07-32_2026_signflip60_fedcap_pathalogical", DataDistribution.Pathological)
    # generalize_test("RUN_Mon_Oct__5_13-38-29_2026_signflip60_fedclipcfl_pathalogical")
    # generalize_test("RUN_Mon_Oct__5_01-09-14_2026_signflip60_fedattract_pathalogical")
    # generalize_test("RUN_Tue_Oct__6_16-07-32_2026_signflip60_fedcap_pathalogical")
    # generalize_test("RUN_Mon_Oct__5_18-14-36_2026_signflip60_fedattract_pathalogical_50rounds")

    # test_labelswitch("RUN_Sun_Oct__4_17-05-19_2026_labelswitch60_fedattract_pathalogical_clip3_norecalc")
    # test_backdoor(folder_to_test)
    # blind_test_backdoor(folder_to_test)
    # blind_test_backdoor_topology(foslder_to_test)
    # evaluate_entropy(folder_to_test)
