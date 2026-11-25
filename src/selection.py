import argparse
from pathlib import Path

import pandas as pd

from src import config

MIN_SEEDS = 5
WORST_SEED_FLOOR = 0.85
SUMMARY_COLUMNS = ["section", "cfg_name", "seeds", "f1_mean", "f1_std", "f1_min", "f1_max", "rainy_f1_mean",
                   "rainy_f1_std", "rainy_f1_min", "epochs_mean", "best_epoch_mean", "minutes_mean"]


def config_rows(results, section=None):
    rows = results[~results["cfg_name"].str.startswith("smoke")]
    return rows if section is None else rows[rows["section"] == section]


def summarize_group(group):
    f1, rainy = group["val_macro_f1"], group["val_f1_rainy"]
    return pd.Series({"seeds": len(group), "f1_mean": f1.mean(), "f1_std": f1.std(ddof=1), "f1_min": f1.min(),
                      "f1_max": f1.max(), "rainy_f1_mean": rainy.mean(), "rainy_f1_std": rainy.std(ddof=1),
                      "rainy_f1_min": rainy.min(), "epochs_mean": group["epochs_run"].mean(),
                      "best_epoch_mean": group["best_epoch"].mean(), "minutes_mean": group["minutes"].mean()})


def config_summary(results, section=None):
    grouped = config_rows(results, section).groupby(["section", "cfg_name"])
    summary = grouped.apply(summarize_group, include_groups=False).reset_index()
    return summary[SUMMARY_COLUMNS].sort_values("f1_mean", ascending=False).reset_index(drop=True)


def eligible(summary, min_seeds=MIN_SEEDS, floor=WORST_SEED_FLOOR):
    return summary[(summary["seeds"] >= min_seeds) & (summary["f1_min"] >= floor)]


def is_tie(first, second):
    return abs(first["f1_mean"] - second["f1_mean"]) < max(first["f1_std"], second["f1_std"])


def select_config(summary, min_seeds=MIN_SEEDS, floor=WORST_SEED_FLOOR):
    candidates = eligible(summary, min_seeds, floor).sort_values("f1_mean", ascending=False)
    if candidates.empty:
        return None
    top = candidates.iloc[0]
    tied = candidates[candidates.apply(lambda row: is_tie(top, row), axis=1)]
    return tied.sort_values(["f1_std", "f1_mean"], ascending=[True, False]).iloc[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, action="append")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = pd.read_csv(Path(config.EXPERIMENTS_DIR) / "results.csv")
    summary = config_summary(results[results["section"].isin(args.section)] if args.section else results)
    print(summary.round(3).to_string(index=False))
    chosen = select_config(summary)
    print("Selected:", "none eligible" if chosen is None else f"{chosen['cfg_name']} (section {chosen['section']})")
