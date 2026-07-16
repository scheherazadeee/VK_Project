"""Точка входа: python -m src.train --config configs/ials.yaml

Один запуск = один конфиг = один results/<run_name>.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.data import dataset, matrix, target
from src.eval import metrics, slices
from src.models.bpr import BPRModel
from src.models.ease import EASEModel
from src.models.hybrid import IALSContentFallback
from src.models.ials import IALSModel
from src.models.knn import ContentKNNModel
from src.models.popularity import PopularityModel
from src.models.two_tower import TwoTowerModel

MODELS = {
    "popularity": PopularityModel,
    "ials": IALSModel,
    "bpr": BPRModel,
    "ease": EASEModel,
    "content_knn": ContentKNNModel,
    "ials_content_fallback": IALSContentFallback,
    "two_tower": TwoTowerModel,
}

# контентным моделям доступны айтемы из eval-недель
CONTENT_MODELS = {"content_knn", "ials_content_fallback"}


def build_eval_sets(eval_pos: pl.DataFrame) -> tuple[np.ndarray, list[set]]:
    grouped = eval_pos.group_by("user_id").agg(pl.col("item_id")).sort("user_id")
    user_ids = grouped["user_id"].to_numpy()
    positives = [set(items) for items in grouped["item_id"].to_list()]
    return user_ids, positives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    cfg["seed"] = seed
    np.random.seed(seed)
    k_max = max(cfg.get("ks", [10, 50]))
    t0 = time.time()

    # данные и таргет
    split = dataset.load_interactions(cfg.get("subsample", "ur0.01_ir0.01"))
    items_meta = dataset.load_items_metadata()
    variant = cfg.get("target", "combined")
    train_pos = target.make_positives(dataset.add_duration(split.train, items_meta), variant)
    eval_inter = split.val if args.eval_split == "val" else split.test
    eval_pos = target.make_positives(dataset.add_duration(eval_inter, items_meta), variant)

    extend = (cfg["model"] in CONTENT_MODELS
              or (cfg["model"] == "two_tower" and cfg.get("params", {}).get("use_emb")))
    extra_items = eval_pos["item_id"].unique().to_numpy() if extend else None
    mapping = matrix.build_mapping(train_pos, extra_item_ids=extra_items)
    weight_col = None
    if cfg.get("confidence_weighting", False):
        train_pos = train_pos.with_columns(target.confidence_expr(cfg.get("alpha", 20.0)))
        weight_col = "confidence"
    train_csr = matrix.to_csr(train_pos, mapping, weight_col=weight_col)
    print(f"train: {train_csr.nnz} positives, {mapping.n_users} users x {mapping.n_items} items")

    eval_user_ids, positives = build_eval_sets(eval_pos)
    print(f"eval ({args.eval_split}): {len(eval_user_ids)} users with positives")

    # обучение
    fit_kwargs: dict = {"weighted": weight_col is not None}
    if extend:
        # выравниваем эмбеддинги с каталогом; айтемы без эмбеддинга остаются нулевыми
        emb_ids, embs = dataset.load_item_embeddings(mapping.item_ids, dim=64)
        item_emb = np.zeros((mapping.n_items, embs.shape[1]), dtype=np.float32)
        pos_in_catalog = np.searchsorted(mapping.item_ids, emb_ids)
        item_emb[pos_in_catalog] = embs
        fit_kwargs["item_emb"] = item_emb
        print(f"embeddings: {len(emb_ids)}/{mapping.n_items} items covered")

    if cfg["model"] == "two_tower" and cfg.get("params", {}).get("use_meta"):
        im = (pl.DataFrame({"item_id": mapping.item_ids})
              .join(items_meta.select("item_id", "duration"), on="item_id", how="left")
              .with_columns((pl.col("duration").fill_null(0) // 10).alias("dur_bucket")))
        fit_kwargs["item_meta"] = im["dur_bucket"].to_numpy().astype(np.int64)
        um = (pl.DataFrame({"user_id": mapping.user_ids})
              .join(dataset.load_users_metadata(), on="user_id", how="left")
              .with_columns((pl.col("age").fill_null(0) // 10).alias("age_bucket"),
                            pl.col("gender").fill_null(0), pl.col("geo").fill_null(0)))
        fit_kwargs["user_meta"] = um.select("age_bucket", "gender", "geo").to_numpy().astype(np.int64)

    model_cls = MODELS[cfg["model"]]
    model_params = cfg.get("params", {})
    if cfg["model"] != "popularity":
        model_params["seed"] = seed
    model = model_cls(**model_params)
    model.fit(train_csr, **fit_kwargs)

    # рекомендации: CF для известных юзеров, popularity-fallback для холодных
    u_idx_map = {u: i for i, u in enumerate(mapping.user_ids)}
    known_mask = np.array([u in u_idx_map for u in eval_user_ids])
    recs_raw = np.full((len(eval_user_ids), k_max), -1, dtype=np.int64)

    known_idx = np.array([u_idx_map[u] for u in eval_user_ids[known_mask]], dtype=np.int64)
    if len(known_idx):
        recs_raw[known_mask] = mapping.item_ids[model.recommend(known_idx, train_csr, k=k_max)]
    if (~known_mask).any():
        pop = PopularityModel().fit(train_csr)
        top = mapping.item_ids[pop.ranking_[:k_max]]
        recs_raw[~known_mask] = top

    # метрики
    ks = tuple(cfg.get("ks", [10, 50]))
    result = {
        "config": cfg, "eval_split": args.eval_split, "seed": seed,
        "overall": metrics.evaluate_topk(recs_raw, positives, ks=ks),
    }
    eval_items = eval_pos["item_id"].unique().to_numpy()
    result["item_slices"] = slices.evaluate_item_slices(
        recs_raw, positives, slices.item_slices(train_pos, eval_items), ks=ks)
    result["user_slices"] = slices.evaluate_user_slices(
        recs_raw, positives, eval_user_ids, slices.user_slices(train_pos, eval_user_ids), ks=ks)

    pop_counts = np.asarray((train_csr > 0).sum(axis=0)).ravel()
    raw_pop = dict(zip(mapping.item_ids.tolist(), pop_counts.tolist()))
    pop_of_recs = np.vectorize(lambda i: raw_pop.get(i, 0))(recs_raw)
    result["beyond_accuracy"] = {
        "coverage@50": len(np.unique(recs_raw[:, :50])) / mapping.n_items,
        "novelty@10": float(
            -np.log2(np.maximum(pop_of_recs[:, :10], 1) / max(train_csr.nnz, 1)).mean()),
    }
    result["runtime_sec"] = round(time.time() - t0, 1)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    run_name = cfg.get("run_name", f"{cfg['model']}_{variant}")
    if args.seed is not None:
        run_name = f"{run_name}_seed{seed}"
    out_path = out_dir / f"{run_name}_{args.eval_split}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["overall"], indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
