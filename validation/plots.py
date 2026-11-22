"""Static matplotlib figures from the validation report (for the README / notebook).

    python -m validation.plots
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from training.config import FIG_DIR, RESULTS_DIR  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
NAMES = {"median_baseline": "Median baseline", "condition_median": "Condition median", "gamma_glm": "Gamma GLM",
         "xgboost": "XGBoost", "xgboost_aft": "XGBoost AFT", "catboost": "CatBoost", "ft_transformer": "FT-Transformer"}


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e7e6e2", lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    r = json.loads((RESULTS_DIR / "validation_report.json").read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(r["nested_cv"])
    T = r["test"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, k in enumerate(keys):
        b = T["bootstrap"][k]["metrics"]["mae"]
        axes[0].plot(b["ci_bca"], [i, i], color=BLUE, lw=2)
        axes[0].plot(b["estimate"], i, "o", color=BLUE)
        v = r["nested_cv"][k]["summary"]["mae"]["values"]
        axes[1].plot(v, np.full(len(v), i) + np.linspace(-.12, .12, len(v)), "o", color=BLUE, alpha=.6)
    for ax in axes:
        ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
        ax.invert_yaxis()
    _style(axes[0], "Test MAE with 95% BCa bootstrap CI", "MAE (days)")
    _style(axes[1], "Nested-CV outer-fold MAE", "MAE (days)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison.png", dpi=150); plt.close(fig)

    perm = r["permutation_tests"]
    fig, axes = plt.subplots(1, len(perm), figsize=(3.2 * len(perm), 3), sharey=False)
    for ax, (k, p) in zip(np.atleast_1d(axes), perm.items()):
        ax.hist(p["null_mae"], bins=15, color=GREY, alpha=.6)
        ax.axvline(p["observed_mae"], color=BLUE, lw=2)
        _style(ax, f"{NAMES[k]}\np = {p['p_value']:.3f}", "val. MAE")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shuffled_target.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    k = r["selection"]["display_model"]
    c = T["calibration"][k]["quantile_bins"]
    axes[0].plot([b["mean_predicted"] for b in c], [b["mean_observed"] for b in c], "o-", color=BLUE)
    lo, hi = 1, 30
    axes[0].plot([lo, hi], [lo, hi], "--", color=GREY); axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
    _style(axes[0], f"Calibration ({NAMES[k]})", "Mean predicted LOS", "Mean observed LOS")
    lc = r["learning_curves"]["catboost"]["points"]
    axes[1].plot([p["fraction"] for p in lc], [p["train_mae"] for p in lc], "o--", color=ORANGE, label="Training")
    axes[1].plot([p["fraction"] for p in lc], [p["val_mae"] for p in lc], "o-", color=BLUE, label="Validation")
    axes[1].legend(frameon=False); _style(axes[1], "Learning curve (CatBoost)", "Share of training data", "MAE")
    lv = T["conformal"][k]["levels"]
    axes[2].plot([v["nominal"] for v in lv.values()], [v["coverage"] for v in lv.values()], "o-", color=BLUE)
    axes[2].plot([.5, .95], [.5, .95], "--", color=GREY)
    _style(axes[2], "Conformal interval coverage", "Nominal", "Empirical (test)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration_learning_coverage.png", dpi=150); plt.close(fig)

    h = r["data"]["los_histogram"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar([int(x) for x in h], list(h.values()), color=BLUE, width=.8)
    _style(ax, "Length of stay distribution (synthetic data)", "Days", "Patients")
    fig.tight_layout(); fig.savefig(FIG_DIR / "los_distribution.png", dpi=150); plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
