import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.ticker import MaxNLocator
import threading
from typing import List, Optional

def analyze_feature_similarity(
    full_features: np.ndarray,
    client_indices: List[List[int]],
    distance_metric: str = 'euclidean',
    linkage_method: str = 'average',
    plot_heatmap: bool = True,
    plot_dendrogram: bool = True,
    save_prefix: Optional[str] = None,
    run_async: bool = False
) -> Optional[threading.Thread]:
    """
    Analyze client similarity based on feature centroids.
    """
    def _run():
        # Compute centroids
        centroids = []
        for idxs in client_indices:
            if len(idxs) > 0:
                centroid = full_features[idxs].mean(axis=0)
            else:
                centroid = np.zeros(full_features.shape[1])
            centroids.append(centroid)
        centroids = np.array(centroids)
        n_clients = len(centroids)
        
        # Compute pairwise distance matrix
        if distance_metric == 'cosine':
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms[norms == 0] = 1
            centroids_norm = centroids / norms
            sim = centroids_norm @ centroids_norm.T
            dist_matrix = 1 - sim
        elif distance_metric == 'euclidean':
            dist_matrix = squareform(pdist(centroids, metric='euclidean'))
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Ensure diagonal is exactly zero (fix floating point errors)
        np.fill_diagonal(dist_matrix, 0.0)
        
        # Plot heatmap (using similarity = 1 - distance for cosine, or negative distance for euclidean)
        if plot_heatmap:
            plt.figure(figsize=(12, 10))
            if distance_metric == 'cosine':
                similarity = 1 - dist_matrix
            else:
                # For euclidean, convert to similarity using negative distance (or inverse)
                similarity = -dist_matrix
            sns.heatmap(similarity, cmap='viridis', annot=False, square=True,
                        xticklabels=[f"C{i}" for i in range(n_clients)],
                        yticklabels=[f"C{i}" for i in range(n_clients)],
                        cbar_kws={'label': 'Similarity'})
            plt.title(f'Client Similarity (1 - {distance_metric} distance)')
            if save_prefix:
                plt.savefig(f"{save_prefix}_heatmap.png", dpi=150, bbox_inches='tight')
            plt.show()
        
        # Hierarchical clustering dendrogram
        if plot_dendrogram:
            # Convert distance matrix to condensed form
            # For cosine, we already have square matrix; for euclidean, we have square as well.
            # Use squareform with checks disabled or just extract upper triangle.
            condensed = squareform(dist_matrix, checks=False)  # checks=False avoids diagonal check
            Z = linkage(condensed, method=linkage_method)
            
            plt.figure(figsize=(14, 7))
            # Generate labels
            labels = [f"Client {i}" for i in range(n_clients)]
            # Plot dendrogram with rotated labels
            dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)
            plt.title(f'Client Hierarchy (linkage={linkage_method}, metric={distance_metric})')
            plt.xlabel('Client')
            plt.ylabel('Distance')
            plt.tight_layout()
            if save_prefix:
                plt.savefig(f"{save_prefix}_dendrogram.png", dpi=150, bbox_inches='tight')
            plt.show()
        
        # Additional metrics
        intra_var = []
        for idxs in client_indices:
            if len(idxs) > 1:
                feat_client = full_features[idxs]
                centroid = centroids[len(intra_var)]
                var = np.mean(np.sum((feat_client - centroid)**2, axis=1))
                intra_var.append(var)
            else:
                intra_var.append(0.0)
        
        triu_indices = np.triu_indices(n_clients, k=1)
        inter_distances = dist_matrix[triu_indices]
        
        print("[Analyzer] Feature similarity results:")
        print(f"  Distance metric: {distance_metric}")
        print(f"  Mean inter‑client distance: {np.mean(inter_distances):.4f} ± {np.std(inter_distances):.4f}")
        print(f"  Min inter‑client distance: {np.min(inter_distances):.4f}")
        print(f"  Max inter‑client distance: {np.max(inter_distances):.4f}")
        print(f"  Mean intra‑client variance (first 5): {np.mean(intra_var[:5]):.4f}")
        print("[Analyzer] Done.")
    
    if run_async:
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
    else:
        _run()
        return None