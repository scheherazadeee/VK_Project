# графики EDA для отчёта
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import dataset, target  # noqa: E402

FIG_DIR = Path("reports/figures")
STYLE = {"figure.figsize": (7, 4.2), "figure.dpi": 130, "axes.grid": True,
         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False}
COLOR = "#3b6ea5"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR / name}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    split = dataset.load_interactions()
    items_meta = dataset.load_items_metadata()
    train = target.with_watch_ratio(dataset.add_duration(split.train, items_meta))

    print(f"train interactions: {len(train):,}")
    print(f"users: {train['user_id'].n_unique():,}, items: {train['item_id'].n_unique():,}")

    # 1. long-tail item popularity (log-log)
    counts = train.group_by("item_id").len().sort("len", descending=True)["len"].to_numpy()
    fig, ax = plt.subplots()
    ax.loglog(np.arange(1, len(counts) + 1), counts, color=COLOR, lw=1.5)
    ax.set_xlabel("Item popularity rank")
    ax.set_ylabel("Interactions in train")
    ax.set_title("Item popularity distribution (long tail)")
    save(fig, "01_item_longtail.png")

    # 2. watch-ratio distribution
    wr = train["watch_ratio"].drop_nulls().to_numpy()
    fig, ax = plt.subplots()
    ax.hist(wr, bins=50, color=COLOR, alpha=0.85)
    ax.set_xlabel("watch ratio = timespent / duration")
    ax.set_ylabel("Interactions")
    ax.set_title("Distribution of watch ratio (train)")
    save(fig, "02_watch_ratio.png")

    # 3. duration distribution
    dur = train["duration"].drop_nulls().to_numpy()
    fig, ax = plt.subplots()
    ax.hist(dur, bins=60, color=COLOR, alpha=0.85)
    ax.set_xlabel("Video duration, s")
    ax.set_ylabel("Interactions")
    ax.set_title("Distribution of video duration")
    save(fig, "03_duration.png")

    # 4. weekly activity
    weekly = split.train.group_by("week").len().sort("week")
    fig, ax = plt.subplots()
    ax.bar(weekly["week"].to_numpy(), weekly["len"].to_numpy(), color=COLOR, alpha=0.85)
    ax.set_xlabel("Week")
    ax.set_ylabel("Interactions")
    ax.set_title("Weekly activity (train)")
    save(fig, "04_weekly_activity.png")

    # 5. доля позитивов по вариантам таргета + аномалии
    print("\nTarget variants (train):")
    for variant in target.TARGET_VARIANTS:
        share = train.select(target.positive_expr(variant).fill_null(False).mean()).item()
        print(f"  {variant:9s}: positives share = {share:.4f}")
    anom = train.filter(pl.col("watch_ratio") == 1.0).height
    no_dur = train.filter(pl.col("duration").is_null() | (pl.col("duration") == 0)).height
    print(f"\nanomalies: watch_ratio clipped at 1.0 (timespent >= duration): {anom:,}")
    print(f"missing/zero duration: {no_dur:,}")


if __name__ == "__main__":
    main()
