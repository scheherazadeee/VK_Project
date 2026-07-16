"""iALS + контентный fallback для холодных айтемов.

Warm-айтемы ранжирует iALS. Cold-айтемам отводится фиксированная квота позиций
в выдаче, они отбираются по контентной близости к профилю пользователя.
"""

import numpy as np
import scipy.sparse as sp

from .ials import IALSModel
from .knn import _l2norm


class IALSContentFallback:
    name = "ials_content_fallback"

    def __init__(self, factors: int = 128, regularization: float = 0.01,
                 alpha: float = 20.0, iterations: int = 20,
                 dim: int = 64, cold_share: float = 0.1, seed: int = 42):
        self.ials = IALSModel(factors=factors, regularization=regularization,
                              alpha=alpha, iterations=iterations, seed=seed)
        self.dim = dim
        self.cold_share = cold_share

    def fit(self, train_csr: sp.csr_matrix, **kwargs) -> "IALSContentFallback":
        self.ials.fit(train_csr, **kwargs)
        emb = kwargs["item_emb"][:, : self.dim].astype(np.float32)
        self.item_emb_ = _l2norm(emb)

        counts = np.asarray((train_csr > 0).sum(axis=0)).ravel()
        has_emb = np.linalg.norm(emb, axis=1) > 0
        self.cold_ = np.where((counts == 0) & has_emb)[0]
        self.cold_emb_ = self.item_emb_[self.cold_]
        return self

    def recommend(self, user_indices: np.ndarray, train_csr: sp.csr_matrix,
                  k: int = 100, batch: int = 4096) -> np.ndarray:
        n_cold_slots = max(1, int(round(k * self.cold_share)))
        n_warm_slots = k - n_cold_slots
        # cold-позиции равномерно разбросаны по выдаче
        cold_pos = np.linspace(2, k - 1, n_cold_slots).round().astype(int)
        is_cold_pos = np.zeros(k, dtype=bool)
        is_cold_pos[cold_pos] = True

        warm_recs = self.ials.recommend(user_indices, train_csr, k=n_warm_slots)

        recs = np.empty((len(user_indices), k), dtype=np.int64)
        for s in range(0, len(user_indices), batch):
            idx = user_indices[s:s + batch]
            hist = train_csr[idx]
            profiles = _l2norm(np.asarray(hist @ self.item_emb_))
            cold_scores = profiles @ self.cold_emb_.T
            top = np.argpartition(-cold_scores, n_cold_slots, axis=1)[:, :n_cold_slots]
            order = np.take_along_axis(cold_scores, top, axis=1).argsort(axis=1)[:, ::-1]
            cold_recs = self.cold_[np.take_along_axis(top, order, axis=1)]

            block = np.empty((len(idx), k), dtype=np.int64)
            block[:, is_cold_pos] = cold_recs
            block[:, ~is_cold_pos] = warm_recs[s:s + batch]
            recs[s:s + batch] = block
        return recs
