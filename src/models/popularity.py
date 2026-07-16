"""Popularity baseline: топ-N айтемов по числу позитивов в train."""

import numpy as np
import scipy.sparse as sp


class PopularityModel:
    name = "popularity"

    def fit(self, train_csr: sp.csr_matrix, **kwargs) -> "PopularityModel":
        pop = np.asarray((train_csr > 0).sum(axis=0)).ravel()
        self.ranking_ = np.argsort(-pop)
        return self

    def recommend(self, user_indices: np.ndarray, train_csr: sp.csr_matrix,
                  k: int = 100) -> np.ndarray:
        recs = np.empty((len(user_indices), k), dtype=np.int64)
        for row, u in enumerate(user_indices):
            seen = set(train_csr.indices[train_csr.indptr[u]:train_csr.indptr[u + 1]])
            out = [i for i in self.ranking_[: k + len(seen)] if i not in seen]
            recs[row] = out[:k]
        return recs
