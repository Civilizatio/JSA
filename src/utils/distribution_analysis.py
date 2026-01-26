# src/utils/distribution_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def normalize_counter(counter: np.ndarray, eps=1e-12):
    p = counter.astype(np.float64)
    p = p + eps
    return p / p.sum()


def js_divergence_matrix(P: np.ndarray):
    """
    P: [N, K] probability matrix
    return: [N, N] JS divergence
    """
    N = P.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            d = jensenshannon(P[i], P[j]) ** 2
            D[i, j] = D[j, i] = d
    return D


def topk_overlap_matrix(P: np.ndarray, k=50):
    """
    P: [N, K]
    """
    N = P.shape[0]
    topk = np.argsort(-P, axis=1)[:, :k]
    O = np.zeros((N, N))
    for i in range(N):
        set_i = set(topk[i])
        for j in range(N):
            set_j = set(topk[j])
            O[i, j] = len(set_i & set_j) / k
    return O


def plot_heatmap(mat, title, save_path, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_embedding(
    emb: np.ndarray,
    labels=None,
    names=None,
    title="Embedding",
    save_path=None,
    annotate=False,
    cmap="tab10",
):
    """
    emb: [N, 2]
    labels: [N,] int or None (for coloring)
    names: list[str] or None (for annotation)
    """
    plt.figure(figsize=(6, 5))

    if labels is not None:
        scatter = plt.scatter(
            emb[:, 0],
            emb[:, 1],
            c=labels,
            cmap=cmap,
            s=30,
        )
        plt.colorbar(scatter, fraction=0.046, pad=0.04)
    else:
        plt.scatter(emb[:, 0], emb[:, 1], s=30)

    if annotate and names is not None:
        for i, name in enumerate(names):
            plt.text(
                emb[i, 0],
                emb[i, 1],
                str(name),
                fontsize=8,
                alpha=0.8,
            )

    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def pca_embedding(P, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(P), pca.explained_variance_ratio_


def umap_embedding(P, n_components=2, random_state=42):
    if not HAS_UMAP:
        raise ImportError("UMAP not installed")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(P)
