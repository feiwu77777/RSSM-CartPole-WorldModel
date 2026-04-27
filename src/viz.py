"""Video and figure helpers for posterior reconstruction, imagined rollouts, and latent PCA."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skvideo.io
from sklearn.decomposition import PCA


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """frame: (3, H, W) float in [0,1] -> (H, W, 3) uint8"""
    f = np.clip(frame, 0.0, 1.0)
    f = np.transpose(f, (1, 2, 0))
    return (f * 255.0).astype(np.uint8)


def write_side_by_side_video(
    truth: np.ndarray,
    pred: np.ndarray,
    path: str,
    fps: int = 15,
) -> None:
    """truth, pred: (T, 3, H, W) float arrays in [0, 1]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    T = truth.shape[0]
    frames = []
    for t in range(T):
        left = _to_uint8(truth[t])
        right = _to_uint8(pred[t])
        gap = np.zeros((left.shape[0], 4, 3), dtype=np.uint8)
        frames.append(np.concatenate([left, gap, right], axis=1))
    video = np.stack(frames, axis=0)
    skvideo.io.vwrite(path, video, inputdict={"-r": str(fps)}, outputdict={"-r": str(fps)})


def plot_latent_pca(
    features: np.ndarray,
    color_values: np.ndarray,
    path: str,
    title: str = "Latent space (PCA, colored by pole angle)",
) -> None:
    """features: (N, D); color_values: (N,) — typically pole angle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_values, cmap="viridis", s=4)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.colorbar(sc, ax=ax, label="pole angle (rad)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
