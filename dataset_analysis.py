import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


def analyze_splits_async(dataset, splits, num_classes=10):
    """
    Runs dataset distribution analysis in a background thread.

    Args:
        dataset: PyTorch dataset
        splits: list of (train_indices, test_indices)
        num_classes: number of classes
        save_dir: where to save outputs
    """

    def _run():
        print("[Analyzer] Starting...")

        # os.makedirs(save_dir, exist_ok=True)

        # -------------------------
        # Fast label access (important optimization)
        # -------------------------
        if hasattr(dataset, "targets"):
            all_labels = np.array(dataset.targets)
            get_labels = lambda idx: all_labels[idx]
        else:
            get_labels = lambda idx: np.array([dataset[i][1] for i in idx])

        # -------------------------
        # Compute distributions
        # -------------------------
        distributions = []
        for train_idx, _ in splits:
            labels = get_labels(train_idx)
            count = np.bincount(labels, minlength=num_classes)
            prob = count / max(count.sum(), 1)
            distributions.append(prob)

        distributions = np.array(distributions)

        # -------------------------
        # Plot label distribution
        # -------------------------
        plt.figure()
        plt.imshow(distributions, aspect='auto')
        plt.colorbar()
        plt.xlabel("Class")
        plt.ylabel("Client")
        plt.title("Label Distribution per Client")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig(os.path.join(save_dir, "label_distribution.png"))
        plt.show()

        # -------------------------
        # Compute distance matrix
        # -------------------------
        n = len(distributions)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = jensenshannon(
                    distributions[i], distributions[j]
                )

        # -------------------------
        # Plot similarity
        # -------------------------
        plt.figure()
        plt.imshow(dist_matrix)
        plt.colorbar()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Client Similarity (JS Distance)")
        # plt.savefig(os.path.join(save_dir, "client_similarity.png"))
        plt.show()

        # -------------------------
        # Metrics
        # -------------------------
        global_dist = distributions.mean(axis=0)

        entropies = np.array([entropy(p) for p in distributions])
        non_iid_scores = np.array([
            jensenshannon(p, global_dist) for p in distributions
        ])

        # -------------------------
        # Save everything
        # -------------------------
        # np.save(os.path.join(save_dir, "distributions.npy"), distributions)
        # np.save(os.path.join(save_dir, "distance_matrix.npy"), dist_matrix)
        # np.save(os.path.join(save_dir, "entropy.npy"), entropies)
        # np.save(os.path.join(save_dir, "non_iid_score.npy"), non_iid_scores)

        print("[Analyzer] Done.")
        print("Entropy (first 5):", entropies[:5])
        print("Non-IID score (first 5):", non_iid_scores[:5])

    # -------------------------
    # Run in background thread
    # -------------------------
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return thread