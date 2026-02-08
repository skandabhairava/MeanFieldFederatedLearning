import logging as log

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
        Client(i, client_splits, types[i])
        for i in range(config.NUM_CLIENTS)
    ]

    log.debug("Created Clients")

    model = models.get_model()

    log.debug("Model loaded.")

    server = Server(model, clients, config.RANDOM_SEED)

    log.info("Starting Training")
    server.train(combined_data, "20_byzantine_flip")


if __name__ == "__main__":
    main()
