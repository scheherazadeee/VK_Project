"""EASE (Steck 2019), закрытая форма: B = -P / diag(P), P = (X^T X + lambda I)^-1.

Плотная матрица айтем-айтем, поэтому запускается только на плотном сабсэмпле.
"""

import numpy as np
import scipy.sparse as sp


class EASEModel:
    name = "ease"

    def __init__(self, l2: float = 200.0, max_items: int = 30000, seed: int = 42):
        self.l2 = l2
        self.max_items = max_items

    def fit(self, train_csr: sp.csr_matrix, **kwargs) -> "EASEModel":
        n_items = train_csr.shape[1]
        if n_items > self.max_items:
            raise MemoryError(
                f"EASE needs a dense {n_items}x{n_items} matrix "
                f"({n_items**2 * 4 / 1e9:.1f} GB)")
        X = train_csr.astype(np.float32)
        G = np.asarray((X.T @ X).todense())
        G[np.diag_indices(n_items)] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        self.B_ = B
        return self

    def recommend(self, user_indices: np.ndarray, train_csr: sp.csr_matrix,
                  k: int = 100, batch: int = 4096) -> np.ndarray:
        recs = np.empty((len(user_indices), k), dtype=np.int64)
        for s in range(0, len(user_indices), batch):
            idx = user_indices[s:s + batch]
            hist = train_csr[idx]
            scores = np.asarray(hist @ self.B_)
            scores[hist.nonzero()] = -np.inf
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
            recs[s:s + batch] = np.take_along_axis(top, order, axis=1)
        return recs
