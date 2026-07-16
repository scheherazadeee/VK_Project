# итоговая таблица по test: mean +/- std по сидам
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["recall@10", "ndcg@10", "recall@50", "ndcg@50", "map@10"]
SLICED = [("cold_items", "ndcg@10"), ("warm_items", "ndcg@10")]


def collect() -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for f in Path("results").glob("*_test.json"):
        base = re.sub(r"_seed\d+$", "", f.stem.removesuffix("_test"))
        groups[base].append(json.loads(f.read_text()))
    return groups


def fmt(vals: list[float]) -> str:
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def main() -> None:
    groups = collect()
    header = (["run", "n_seeds"] + METRICS
              + [f"{s}_{m}" for s, m in SLICED] + ["coverage@50", "novelty@10"])
    csv_lines = [",".join(header)]
    md_lines = ["| " + " | ".join(header) + " |",
                "|" + "---|" * len(header)]

    for name in sorted(groups):
        runs = groups[name]
        row = [name, str(len(runs))]
        for m in METRICS:
            row.append(fmt([r["overall"][m] for r in runs]))
        for s, m in SLICED:
            row.append(fmt([r["item_slices"][s][m] for r in runs]))
        row.append(fmt([r["beyond_accuracy"]["coverage@50"] for r in runs]))
        row.append(fmt([r["beyond_accuracy"]["novelty@10"] for r in runs]))
        csv_lines.append(",".join(v.replace(",", ";") for v in row))
        md_lines.append("| " + " | ".join(row) + " |")

    Path("results/final_test_table.csv").write_text("\n".join(csv_lines) + "\n")
    Path("results/final_test_table.md").write_text("\n".join(md_lines) + "\n")
    print(f"{len(groups)} model groups")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
