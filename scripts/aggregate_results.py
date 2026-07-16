# собирает результаты в summary.csv и строит графики
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIG_DIR = Path("reports/figures")
STYLE = {"figure.figsize": (7, 4.2), "figure.dpi": 130, "axes.grid": True,
         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False}
C_MAIN, C_COLD, C_WARM, C_POP = "#3b6ea5", "#c44e52", "#dd8452", "#55a868"


def load_runs(split: str) -> dict[str, dict]:
    runs = {}
    for f in sorted(Path("results").glob(f"*_{split}.json")):
        runs[f.stem.removesuffix(f"_{split}")] = json.loads(f.read_text())
    return runs


def summary_csv(runs: dict, split: str) -> None:
    cols = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "map@10"]
    lines = ["run,subsample," + ",".join(cols)
             + ",cold_ndcg@10,warm_ndcg@10,popular_ndcg@10,coverage@50,novelty@10"]
    for name, r in runs.items():
        o = r["overall"]
        s = r["item_slices"]
        b = r["beyond_accuracy"]
        vals = [f"{o[c]:.5f}" for c in cols]
        vals += [f"{s['cold_items']['ndcg@10']:.5f}", f"{s['warm_items']['ndcg@10']:.5f}",
                 f"{s['popular_items']['ndcg@10']:.5f}",
                 f"{b['coverage@50']:.4f}", f"{b['novelty@10']:.2f}"]
        lines.append(f"{name},{r['config'].get('subsample','')}," + ",".join(vals))
    out = Path(f"results/summary_{split}.csv")
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out} ({len(runs)} runs)")


def curve_by_d(runs: dict, prefix: str, metric_path: tuple, label: str):
    ds, ys = [], []
    for d in [4, 8, 16, 32, 64]:
        r = runs.get(f"{prefix}{d}")
        if r is None:
            continue
        node = r
        for k in metric_path:
            node = node[k]
        ds.append(d)
        ys.append(node)
    return ds, ys


def fig_ndcg_vs_d(runs: dict, prefix: str, title: str, fname: str) -> None:
    # главный график: NDCG@10 vs d, срезы cold/warm
    fig, ax = plt.subplots()
    for path, color, label in [
        (("overall", "ndcg@10"), C_MAIN, "All items"),
        (("item_slices", "cold_items", "ndcg@10"), C_COLD, "Cold items"),
        (("item_slices", "warm_items", "ndcg@10"), C_WARM, "Warm items"),
    ]:
        ds, ys = curve_by_d(runs, prefix, path, label)
        if ds:
            ax.plot(ds, ys, "o-", color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64], [4, 8, 16, 32, 64])
    ax.set_xlabel("Content embedding dimensionality d")
    ax.set_ylabel("NDCG@10")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / fname}")


def fig_model_comparison(runs: dict, names: list[str], labels: list[str],
                         fname: str, metric: str = "ndcg@10") -> None:
    present = [(l, runs[n]) for n, l in zip(names, labels) if n in runs]
    if not present:
        return
    labels_, rs = zip(*present)
    x = np.arange(len(rs))
    overall = [r["overall"][metric] for r in rs]
    cold = [r["item_slices"]["cold_items"][metric] for r in rs]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, overall, 0.4, color=C_MAIN, label="All items")
    ax.bar(x + 0.2, cold, 0.4, color=C_COLD, label="Cold items")
    ax.set_xticks(x, labels_, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model comparison: {metric.upper()} (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / fname}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    runs = load_runs(args.split)
    summary_csv(runs, args.split)

    fig_ndcg_vs_d(runs, "content_knn_d", "Content kNN: quality vs signal width",
                  "05_knn_ndcg_vs_d.png")
    fig_ndcg_vs_d(runs, "ials_fallback_d", "iALS + content fallback: quality vs d",
                  "06_hybrid_ndcg_vs_d.png")
    fig_ndcg_vs_d(runs, "tt_emb_d", "Two-tower: quality vs d",
                  "07_tt_ndcg_vs_d.png")

    fig_model_comparison(
        runs,
        ["popularity_combined", "content_knn_d64", "bpr_combined", "ials_combined",
         "ials_fallback_d64", "tt_id_only", "tt_emb_d64", "tt_full"],
        ["Popularity", "Content kNN", "BPR", "iALS",
         "iALS+fallback", "TT id-only", "TT +emb", "TT full"],
        "08_model_comparison.png")


if __name__ == "__main__":
    main()
