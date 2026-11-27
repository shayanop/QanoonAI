import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


def load_metric_values(json_files, metric):
    values = []
    for fp in json_files:
        try:
            data = json.loads(fp.read_text())
            summary = data.get("summary", {})
            if metric in summary:
                values.append(summary[metric])
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}", file=sys.stderr)
    return values


def compute_stats(values):
    if not values:
        return 0.0, 0.0
    avg = mean(values)
    std = stdev(values) if len(values) >= 2 else 0.0
    return avg, std


def main():
    parser = argparse.ArgumentParser(description="Create simple dashboards for QanoonArena results")
    parser.add_argument("--group", default="G10", help="Group id (e.g., G10)")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], help="Splits to plot")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    dashboard_dir = repo_root / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    configs = ["M0+A0", "M1+A0", "M0+A2", "M1+A2"]
    metrics = [
        ("score", "Composite Score", "Score (0-100)"),
        ("tokens_per_turn", "Tokens per Turn", "Tokens"),
        ("latency_p50_ms", "Latency p50", "Latency (ms)"),
        ("ungrounded_rate", "Ungrounded Citation Rate", "Rate"),
    ]

    for split in args.splits:
        # collect json files per config
        all_files_exist = False
        per_config_files = {}
        for cfg in configs:
            pattern = f"{args.group}_{cfg}_{split}_s*.json"
            files = sorted(results_dir.glob(pattern))
            per_config_files[cfg] = files
            if files:
                all_files_exist = True
        if not all_files_exist:
            print(f"No result files found for split '{split}'. Skipping.")
            continue

        # compute means and stds for each metric across configs
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flat
        for ax, (key, title, ylabel) in zip(axes, metrics):
            means = []
            stds = []
            for cfg in configs:
                values = load_metric_values(per_config_files[cfg], key)
                avg, sd = compute_stats(values)
                means.append(avg)
                stds.append(sd)
            # plot bars
            bars = ax.bar(configs, means, yerr=stds, capsize=5, color="skyblue")
            for bar, val in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=9)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.yaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha="right")
        fig.suptitle(f"QanoonArena — {args.group} ({split})", fontsize=14)
        out_path = dashboard_dir / f"dashboard_{split}.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved dashboard for split '{split}' to {out_path}")

if __name__ == "__main__":
    main()
