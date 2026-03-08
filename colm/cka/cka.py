"""
Centered Kernel Alignment (CKA) for comparing representations.

Used to measure how much prosodic information is preserved or lost across layers
(encoder, projector, LLM) by comparing representations of same-text different-prosody
utterance pairs. High CKA = similar geometry; low CKA = prosody-specific structure lost.
"""
from __future__ import annotations

import numpy as np
from typing import Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _center(X: np.ndarray) -> np.ndarray:
    """Center columns."""
    return X - X.mean(axis=0, keepdims=True)


def _center_torch(X: "torch.Tensor") -> "torch.Tensor":
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: np.ndarray, Y: np.ndarray, use_torch: bool = False) -> float:
    """
    Linear CKA between two representation matrices X (n x d1) and Y (n x d2).
    Same n (number of samples); dimensions can differ.
    Returns value in [0, 1]; 1 = identical structure up to linear transform.
    """
    if use_torch and HAS_TORCH and (isinstance(X, torch.Tensor) or isinstance(Y, torch.Tensor)):
        X = torch.as_tensor(X) if not isinstance(X, torch.Tensor) else X
        Y = torch.as_tensor(Y) if not isinstance(Y, torch.Tensor) else Y
        X = _center_torch(X)
        Y = _center_torch(Y)
        n = X.shape[0]
        cov_xx = (X.T @ X) / n
        cov_yy = (Y.T @ Y) / n
        cov_xy = (X.T @ Y) / n
        cka = (cov_xy ** 2).sum() / (torch.sqrt((cov_xx ** 2).sum() * (cov_yy ** 2).sum()) + 1e-10)
        return float(cka.item())
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = _center(X)
    Y = _center(Y)
    n = X.shape[0]
    cov_xx = (X.T @ X) / n
    cov_yy = (Y.T @ Y) / n
    cov_xy = (X.T @ Y) / n
    cka = np.sum(cov_xy ** 2) / (np.sqrt(np.sum(cov_xx ** 2) * np.sum(cov_yy ** 2)) + 1e-10)
    return float(cka)


def kernel_cka(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float | None = None,
    rbf: bool = True,
) -> float:
    """
    Kernel CKA with RBF kernel (or linear if rbf=False).
    sigma: RBF bandwidth; if None, use median heuristic (median pairwise distance).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    X = _center(X)
    Y = _center(Y)

    if rbf and sigma is None:
        from scipy.spatial.distance import pdist
        d = pdist(X, "sqeuclidean")
        sigma = np.median(np.sqrt(d)) if len(d) else 1.0
        sigma = max(sigma, 1e-6)

    if rbf:
        from scipy.spatial.distance import cdist
        Kx = np.exp(-cdist(X, X, "sqeuclidean") / (2 * sigma ** 2))
        Ky = np.exp(-cdist(Y, Y, "sqeuclidean") / (2 * sigma ** 2))
    else:
        Kx = X @ X.T
        Ky = Y @ Y.T

    # Center kernel matrices
    one_n = np.ones((n, n)) / n
    Kx = Kx - one_n @ Kx - Kx @ one_n + one_n @ Kx @ one_n
    Ky = Ky - one_n @ Ky - Ky @ one_n + one_n @ Ky @ one_n

    cka = np.sum(Kx * Ky) / (np.sqrt(np.sum(Kx ** 2) * np.sum(Ky ** 2)) + 1e-10)
    return float(cka)


def batch_linear_cka(
    reprs_a: list[np.ndarray],
    reprs_b: list[np.ndarray],
    aggregate: str = "mean",
) -> np.ndarray | float:
    """
    Compute linear CKA for each pair (reprs_a[i], reprs_b[i]) and optionally aggregate.
    reprs_a, reprs_b: lists of (n_i, d) arrays (variable n_i per pair).
    aggregate: "mean", "median", or "none" (return array of per-pair CKAs).
    """
    ckas = []
    for X, Y in zip(reprs_a, reprs_b):
        X, Y = np.asarray(X), np.asarray(Y)
        n = min(X.shape[0], Y.shape[0])
        if n < 2:
            ckas.append(np.nan)
            continue
        X, Y = X[:n], Y[:n]
        ckas.append(linear_cka(X, Y))
    ckas = np.array(ckas)
    if aggregate == "mean":
        return float(np.nanmean(ckas))
    if aggregate == "median":
        return float(np.nanmedian(ckas))
    return ckas
