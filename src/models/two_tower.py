# two-tower на PyTorch (CPU)
# башня айтема: id-эмбеддинг + замороженный контентный эмбеддинг с проекцией + длительность
# башня юзера: id-эмбеддинг и демография
# лосс InfoNCE с in-batch и hard-негативами

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, cfg: dict,
                 item_content: np.ndarray | None,
                 item_meta: np.ndarray | None, user_meta: np.ndarray | None):
        super().__init__()
        dim = cfg.get("tower_dim", 64)
        id_dim = cfg.get("id_dim", 64)
        hidden = cfg.get("hidden", 128)

        self.user_id_emb = nn.Embedding(n_users, id_dim)
        self.item_id_emb = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.user_id_emb.weight, std=0.05)
        nn.init.normal_(self.item_id_emb.weight, std=0.05)

        self.use_emb = item_content is not None
        item_in = id_dim
        if self.use_emb:
            # контентный эмбеддинг заморожен, обучается только проекция поверх него
            self.content = nn.Embedding.from_pretrained(
                torch.from_numpy(item_content), freeze=True)
            proj_dim = cfg.get("content_proj_dim", 32)
            self.content_proj = nn.Linear(item_content.shape[1], proj_dim)
            item_in += proj_dim

        self.use_meta = item_meta is not None
        user_in = id_dim
        if self.use_meta:
            self.dur_emb = nn.Embedding(int(item_meta.max()) + 1, 8)
            self.item_meta = torch.from_numpy(item_meta)
            item_in += 8
            self.age_emb = nn.Embedding(int(user_meta[:, 0].max()) + 1, 8)
            self.gender_emb = nn.Embedding(int(user_meta[:, 1].max()) + 1, 4)
            self.geo_emb = nn.Embedding(int(user_meta[:, 2].max()) + 1, 8)
            self.user_meta = torch.from_numpy(user_meta)
            user_in += 20

        self.item_mlp = _mlp(item_in, dim, hidden)
        self.user_mlp = _mlp(user_in, dim, hidden)

    def item_vec(self, items: torch.Tensor) -> torch.Tensor:
        parts = [self.item_id_emb(items)]
        if self.use_emb:
            parts.append(self.content_proj(self.content(items)))
        if self.use_meta:
            parts.append(self.dur_emb(self.item_meta[items]))
        return F.normalize(self.item_mlp(torch.cat(parts, dim=-1)), dim=-1)

    def user_vec(self, users: torch.Tensor) -> torch.Tensor:
        parts = [self.user_id_emb(users)]
        if self.use_meta:
            m = self.user_meta[users]
            parts += [self.age_emb(m[:, 0]), self.gender_emb(m[:, 1]), self.geo_emb(m[:, 2])]
        return F.normalize(self.user_mlp(torch.cat(parts, dim=-1)), dim=-1)


class TwoTowerModel:
    name = "two_tower"

    def __init__(self, tower_dim: int = 64, id_dim: int = 64, hidden: int = 128,
                 content_proj_dim: int = 32, use_emb: bool = False, emb_dim: int = 64,
                 use_meta: bool = False, epochs: int = 4, batch_size: int = 8192,
                 lr: float = 3e-3, temperature: float = 0.07,
                 n_hard_negatives: int = 16, seed: int = 42):
        self.cfg = dict(tower_dim=tower_dim, id_dim=id_dim, hidden=hidden,
                        content_proj_dim=content_proj_dim)
        self.use_emb = use_emb
        self.emb_dim = emb_dim
        self.use_meta = use_meta
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.n_hard = n_hard_negatives
        self.seed = seed

    def fit(self, train_csr: sp.csr_matrix, **kwargs) -> "TwoTowerModel":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_users, n_items = train_csr.shape

        item_content = None
        if self.use_emb:
            emb = kwargs["item_emb"][:, : self.emb_dim].astype(np.float32)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            item_content = emb / np.maximum(norm, 1e-12)
        item_meta = kwargs.get("item_meta") if self.use_meta else None
        user_meta = kwargs.get("user_meta") if self.use_meta else None

        self.net = TwoTower(n_users, n_items, self.cfg,
                            item_content, item_meta, user_meta)
        opt = torch.optim.Adam(
            [p for p in self.net.parameters() if p.requires_grad], lr=self.lr)

        coo = train_csr.tocoo()
        users_all = coo.row.astype(np.int64)
        items_all = coo.col.astype(np.int64)

        # hard-негативы сэмплируются по популярности
        pop = np.asarray((train_csr > 0).sum(axis=0)).ravel().astype(np.float64)
        pop_p = pop / pop.sum()

        n = len(users_all)
        steps = int(np.ceil(n / self.batch_size))
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            total = 0.0
            for s in range(steps):
                sl = perm[s * self.batch_size:(s + 1) * self.batch_size]
                u = torch.from_numpy(users_all[sl])
                i = torch.from_numpy(items_all[sl])
                hard = torch.from_numpy(
                    rng.choice(n_items, size=self.n_hard, p=pop_p))

                uv = self.net.user_vec(u)
                iv = self.net.item_vec(i)
                hv = self.net.item_vec(hard)
                logits = uv @ torch.cat([iv, hv]).T / self.temperature
                labels = torch.arange(len(sl))
                loss = F.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"  epoch {epoch + 1}/{self.epochs}: loss={total / steps:.4f}", flush=True)

        with torch.no_grad():
            self.item_vecs_ = self.net.item_vec(torch.arange(n_items)).numpy()
        return self

    def recommend(self, user_indices: np.ndarray, train_csr: sp.csr_matrix,
                  k: int = 100, batch: int = 4096) -> np.ndarray:
        recs = np.empty((len(user_indices), k), dtype=np.int64)
        with torch.no_grad():
            for s in range(0, len(user_indices), batch):
                idx = user_indices[s:s + batch]
                uv = self.net.user_vec(torch.from_numpy(idx.astype(np.int64))).numpy()
                scores = uv @ self.item_vecs_.T
                hist = train_csr[idx]
                scores[hist.nonzero()] = -np.inf
                top = np.argpartition(-scores, k, axis=1)[:, :k]
                order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
                recs[s:s + batch] = np.take_along_axis(top, order, axis=1)
        return recs
