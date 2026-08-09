import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
import joblib  # For saving sklearn models
from typing import Optional, Literal

import stats

# ---------- fixed feature extractor ----------
class FeatureExtractor(nn.Module):
    """ResNet18 feature extractor that outputs flat 512-dim vectors."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer, keep everything else including Global Avg Pooling
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)  # Shape: [batch, 512, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch, 512]
        return x

def get_feature_extractor(device):
    """Returns a feature extractor that outputs 512-dim vectors."""
    model = FeatureExtractor()
    model.to(device)
    model.eval()
    return model

# ---------- feature extraction with caching ----------
def extract_features(dataset, model, device, batch_size=128):
    """Extract features from a dataset using a pretrained model."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            feat = model(images).cpu().numpy()
            features.append(feat)
    
    features = np.concatenate(features, axis=0)
    print(f"Extracted features shape: {features.shape}")  # Should be (N, 512)
    return features

def get_cached_features(dataset, cache_name: str, device, force_recompute: bool = False):
    """
    Extract features from a dataset and cache them as a .npy file.
    cache_name: identifier for the cache file (e.g., "full", "test")
    """
    cache_dir = "./cache/features"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.npy")

    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        features = np.load(cache_path)
        print(f"Loaded features shape: {features.shape}")
        return features

    print(f"Extracting features for {cache_name}...")
    model = get_feature_extractor(device)
    features = extract_features(dataset, model, device)
    np.save(cache_path, features)
    print(f"Saved features to {cache_path}")
    return features

# ---------- clustering with caching ----------
def get_cached_clusters(features: np.ndarray, n_clusters: int, force_recompute: bool = False) -> np.ndarray:
    """
    Perform K-means clustering on features and cache the results.
    
    Args:
        features: Feature matrix of shape (n_samples, n_features)
        n_clusters: Number of clusters for K-means
        force_recompute: If True, recompute even if cached
    
    Returns:
        cluster_ids: Array of shape (n_samples,) with cluster assignments
    """
    cache_dir = "./cache/clusters"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"kmeans_{n_clusters}.npy")
    
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached clusters from {cache_path}")
        kmeans_labels = np.load(cache_path)
        print(f"Loaded K-means model with {n_clusters} clusters")
        return kmeans_labels
    
    print(f"Clustering {features.shape[0]} features into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    cluster_ids = kmeans.fit_predict(features)
    
    # Cache the entire kmeans model (includes cluster centers, labels, etc.)
    # joblib.dump(kmeans, cache_path)
    np.save(cache_path, cluster_ids)
    print(f"Saved clusters to {cache_path}")
    
    return cluster_ids

# ---------- Dirichlet split on clusters ----------
def dirichlet_split_on_clusters(cluster_ids: np.ndarray, n_clients: int, alpha: float) -> list[list[int]]:
    """
    cluster_ids: array of length N, each entry is cluster index (0..K-1)
    Returns list of lists of global indices assigned to each client.
    """
    unique_clusters = np.unique(cluster_ids)
    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    client_indices = [[] for _ in range(n_clients)]

    for cl in unique_clusters:
        indices = cluster_to_indices[cl]
        n = len(indices)
        if n == 0:
            continue

        proportions = np.random.dirichlet(np.ones(n_clients) * alpha)
        counts = (proportions * n).astype(int)

        # adjust rounding errors
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

# ---------- main function ----------
def generate_client_splits(
    n_clients: int,
    alpha: float,
    train_test_split_ratio: float = 0.8,
    test_sampling_mode: Literal["same", "gaussian"] = "same",
    n_feature_clusters: int = 50,
    gaussian_sigma: float = 1.0,
    use_feature_splitting: bool = True,
    force_recompute_features: bool = False,
    force_recompute_clusters: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[list[tuple[list[int], list[int]]], Dataset, np.ndarray|None]:
    """
    Generates client splits with caching of features AND clusters.
    
    Args:
        force_recompute_features: Recompute features from scratch (dataset/model change)
        force_recompute_clusters: Recompute clustering (n_feature_clusters change)
    """
    # Load datasets
    tfm = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)

    # ----- 1. Feature splitting (or label splitting fallback) -----
    if use_feature_splitting:
        # Full dataset (train+test) for training indices
        full_dataset = ConcatDataset([train_dataset, test_dataset])

        # Load cached features
        full_features = get_cached_features(full_dataset, "cifar10_full", device, force_recompute_features)
        test_features = get_cached_features(test_dataset, "cifar10_test", device, force_recompute_features)

        # Verify features are 2D
        assert full_features.ndim == 2, f"Features should be 2D, got shape {full_features.shape}"
        assert test_features.ndim == 2, f"Features should be 2D, got shape {test_features.shape}"

        # Load cached clusters (or compute if needed)
        cluster_ids = get_cached_clusters(full_features, n_feature_clusters, force_recompute_clusters)

        # Dirichlet split on clusters (this is fast, no need to cache)
        client_indices = dirichlet_split_on_clusters(cluster_ids, n_clients, alpha)

        # Compute per‑client feature centroids (needed for Gaussian test sampling)
        client_centroids = []
        for idxs in client_indices:
            if len(idxs) > 0:
                centroid = full_features[idxs].mean(axis=0)
            else:
                centroid = np.zeros(full_features.shape[1])
            client_centroids.append(centroid)

    else:
        # Fallback: original label splitting (no features needed)
        train_targets = np.array(train_dataset.targets)
        test_targets = np.array(test_dataset.targets)
        combined_targets = np.concatenate([train_targets, test_targets])
        client_indices = dirichlet_split_on_clusters(combined_targets, n_clients, alpha)
        if test_sampling_mode == "gaussian":
            print("Warning: 'gaussian' test sampling requires feature splitting. Falling back to 'same'.")
            test_sampling_mode = "same"
        client_centroids = [None] * n_clients
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        full_features = None
        test_features = None

    # ----- 2. Build train/test splits per client -----
    client_splits = []

    if test_sampling_mode == "same":
        # Original behaviour: split each client's own data (full_dataset indices)
        if use_feature_splitting:
            # We need labels for per‑class split; get them from the full dataset
            all_labels = []
            for i in range(len(full_dataset)):
                _, lbl = full_dataset[i]
                all_labels.append(lbl)
            all_labels = np.array(all_labels)
        else:
            all_labels = combined_targets

        for client_id in range(n_clients):
            idxs = np.array(client_indices[client_id])
            targets = all_labels[idxs]

            train_idx = []
            test_idx = []

            for cls in np.unique(targets):
                cls_indices = idxs[targets == cls]
                np.random.shuffle(cls_indices)
                split = int(len(cls_indices) * train_test_split_ratio)
                train_idx.extend(cls_indices[:split].tolist())
                test_idx.extend(cls_indices[split:].tolist())

            client_splits.append((train_idx, test_idx))

    else:  # test_sampling_mode == "gaussian"
        # Training: use all indices allocated to the client (from full_dataset)
        # Test: sample from global test set using Gaussian weights
        for client_id in range(n_clients):
            train_idx = client_indices[client_id]   # indices into full_dataset
            centroid = client_centroids[client_id]

            # Compute distances from each test sample to this client's centroid
            diff = test_features - centroid
            sq_dists = np.sum(diff * diff, axis=1)
            weights = np.exp(-sq_dists / (2 * gaussian_sigma * gaussian_sigma))

            # Normalise to probabilities
            probs = weights / weights.sum()

            # Number of test samples (same heuristic as before)
            n_test = int(len(train_idx) * (1 - train_test_split_ratio))
            n_test = max(1, min(n_test, len(test_dataset)))

            # Sample indices from the global test set
            sampled_test_indices = np.random.choice(
                len(test_dataset), size=n_test, replace=False, p=probs
            )
            # Convert to global indices (full_dataset = train_dataset + test_dataset)
            global_test_indices = (len(train_dataset) + sampled_test_indices).tolist()
            client_splits.append((train_idx, global_test_indices))

    # Optional: print average sizes
    train_avg = stats.avg(len(c[0]) for c in client_splits)
    test_avg = stats.avg(len(c[1]) for c in client_splits)
    print(f"{train_avg=} || {test_avg=}")

    full_combined = ConcatDataset([train_dataset, test_dataset])
    # return client_splits, full_combined
    return client_splits, full_combined, full_features