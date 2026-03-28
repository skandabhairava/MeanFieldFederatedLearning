import logging as log
import pickle
import types as typ

import os
import ray

import lib
import config
import data
import models
from client import Client
from server import Server

def main():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    lib.set_all_seeds(config.RANDOM_SEED)
    lib.set_log_level(log.INFO)

    ray.init(num_cpus=config.NUM_CPUS, num_gpus=config.NUM_GPUS)

    # train_loaders, test_loaders = data.generate_federated_dataloaders(config.NUM_CLIENTS, config.DIRICHLET_ALPHA, config.BATCH_SIZE, train_test_split_ratio=0.8)
    client_splits, combined_data = data.generate_client_splits(config.NUM_CLIENTS, config.DIRICHLET_ALPHA, train_test_split_ratio=0.8)

    types = Client.sample_types(config.NUM_CLIENTS, config.CLIENT_TYPES)

    log.debug(f"Sampled clients: {types}")

    clients = [
        Client(i, client_splits, types[i], seed=config.RANDOM_SEED)
        for i in range(config.NUM_CLIENTS)
    ]

    log.debug("Created Clients")

    model = models.get_model()

    log.debug("Model loaded.")

    server = Server(model, clients, config.RANDOM_SEED)

    log.info("Starting Training")
    save_folder = server.train(combined_data, "byzantine_flip")

    with open(f"{save_folder}/metadata.npy", "wb") as f:
        config_file = {key: val for key, val in vars(config).items() if not key.startswith("__") and not isinstance(val, typ.ModuleType)}
        pickle.dump(config_file, f)

if __name__ == "__main__":
    main()
