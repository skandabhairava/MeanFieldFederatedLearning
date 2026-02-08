# from collections import Counter
import logging as log

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import datasets, transforms

from stats import avg

type ClientSplit = tuple[list[int], list[int]]

def get_datasets():
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    return train, test


def get_combined_targets(train: Dataset, test: Dataset):
    train_targets = np.array(train.targets) # pyright: ignore[reportAttributeAccessIssue]
    test_targets = np.array(test.targets) # pyright: ignore[reportAttributeAccessIssue]
    return np.concatenate([train_targets, test_targets])


def dirichlet_split(labels: np.ndarray, n_clients: int, alpha: float) -> list[list[int]]:
    unique_labels = np.unique(labels)
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    client_indices = [[] for _ in range(n_clients)]

    for label in unique_labels:
        indices = label_to_indices[label]
        n = len(indices)
        if n == 0:
            continue

        proportions = np.random.dirichlet(np.ones(n_clients) * alpha)
        counts = (proportions * n).astype(int)

        remaining = n - counts.sum()
        while remaining > 0:
            counts[np.random.randint(0, n_clients)] += 1
            remaining -= 1
        while remaining < 0:
            i = np.random.choice(np.where(counts > 0)[0])
            counts[i] -= 1
            remaining += 1

        np.random.shuffle(indices)

        start = 0
        for client_id in range(n_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(indices[start:end].tolist())
            start = end

    for c in range(n_clients):
        np.random.shuffle(client_indices[c])

    return client_indices


def generate_client_splits(
    n_clients: int,
    alpha: float,
    train_test_split_ratio: float = 0.8,
) -> tuple[list[ClientSplit], Dataset]:
    train, test = get_datasets()

    combined_targets = get_combined_targets(train, test)
    client_all = dirichlet_split(combined_targets, n_clients, alpha)

    client_splits: list[ClientSplit] = []

    for client_id in range(n_clients):
        idxs = np.array(client_all[client_id])
        targets = combined_targets[idxs]

        train_idx: list[int] = []
        test_idx: list[int] = []

        for cls in np.unique(targets):
            cls_indices = idxs[targets == cls]
            np.random.shuffle(cls_indices)

            split = int(len(cls_indices) * train_test_split_ratio)
            train_idx.extend(cls_indices[:split].tolist())
            test_idx.extend(cls_indices[split:].tolist())

        client_splits.append((train_idx, test_idx))

    train_avg = avg(len(client[0]) for client in client_splits)
    test_avg = avg(len(client[1]) for client in client_splits)
    log.info(f"{train_avg=} || {test_avg=}")
    return client_splits, torch.utils.data.ConcatDataset([train, test])

def build_client_loaders(combined: Dataset, split: ClientSplit, batch_size: int, load_train: bool):
    train_idx, test_idx = split

    if load_train:
        loader = DataLoader(
            Subset(combined, train_idx),
            batch_size=batch_size,
            shuffle=True,
        )

    else:
        loader = DataLoader(
            Subset(combined, test_idx),
            batch_size=batch_size,
            shuffle=False,
        )

    return loader