"""iALS (Hu et al. 2008) через библиотеку implicit."""

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class IALSModel:
    name = "ials"

    def __init__(self, factors: int = 128, regularization: float = 0.01,
                 alpha: float = 20.0, iterations: int = 20, seed: int = 42):
        self.alpha = alpha
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization,
            iterations=iterations, random_state=seed,
            use_gpu=False, calculate_training_loss=False,
        )

    def fit(self, train_csr: sp.csr_matrix, **kwargs) -> "IALSModel":
        weighted = kwargs.get("weighted", False)
        conf = train_csr if weighted else train_csr * self.alpha
        self.model.fit(conf, show_progress=False)
        self.train_csr_ = train_csr
        return self

    def recommend(self, user_indices: np.ndarray, train_csr: sp.csr_matrix,
                  k: int = 100) -> np.ndarray:
        ids, _ = self.model.recommend(
            user_indices, train_csr[user_indices],
            N=k, filter_already_liked_items=True,
        )
        return ids.astype(np.int64)
