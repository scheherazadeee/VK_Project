"""Item-kNN по контентному эмбеддингу (чисто контентный baseline)."""

import numpy as np
import scipy.sparse as sp


def _l2norm(x: np.ndarray, axis: int = 1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


class ContentKNNModel:
    name = "content_knn"

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim

    def fit(self, train_csr: sp.csr_matrix, **kwargs) -> "ContentKNNModel":
        emb = kwargs["item_emb"][:, : self.dim]
        self.item_emb_ = _l2norm(emb.astype(np.float32))
        return self

    def recommend(self, user_indices: np.ndarray, train_csr: sp.csr_matrix,
                  k: int = 100, batch: int = 2048) -> np.ndarray:
        recs = np.empty((len(user_indices), k), dtype=np.int64)
        for s in range(0, len(user_indices), batch):
            idx = user_indices[s:s + batch]
            hist = train_csr[idx]
            # профиль пользователя = среднее эмбеддингов его позитивов
            profiles = _l2norm(np.asarray(hist @ self.item_emb_))
            scores = profiles @ self.item_emb_.T
            scores[hist.nonzero()] = -np.inf
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
            recs[s:s + batch] = np.take_along_axis(top, order, axis=1)
        return recs
