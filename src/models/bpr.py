"""BPR-MF (Rendle et al. 2009) через библиотеку implicit."""

import numpy as np
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking


class BPRModel:
    name = "bpr"

    def __init__(self, factors: int = 128, learning_rate: float = 0.05,
                 regularization: float = 0.01, iterations: int = 100, seed: int = 42):
        self.model = BayesianPersonalizedRanking(
            factors=factors, learning_rate=learning_rate,
            regularization=regularization, iterations=iterations,
            random_state=seed, use_gpu=False,
        )

    def fit(self, train_csr: sp.csr_matrix, **kwargs) -> "BPRModel":
        self.model.fit(train_csr, show_progress=False)
        return self

    def recommend(self, user_indices: np.ndarray, train_csr: sp.csr_matrix,
                  k: int = 100) -> np.ndarray:
        ids, _ = self.model.recommend(
            user_indices, train_csr[user_indices],
            N=k, filter_already_liked_items=True,
        )
        return ids.astype(np.int64)
