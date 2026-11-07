"""Figures used in the paper (300 dpi PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .labeling import LABELS

C_TEXT, C_CTX = "#9aa5b1", "#1f4e79"
CLASS_COLORS = {"Monitoring_Crawler": "#9aa5b1", "Normal_candidate": "#5b8db8", "Suspicious": "#c0392b"}
SHORT = ["Monitoring_\nCrawler", "Normal_\ncandidate", "Suspicious"]


def setup(font: str | None = None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
    if font:
        plt.rcParams["font.family"] = font


def _bar_labels(ax, bars, fmt="{:,.0f}", dy=0.0, size=8):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + dy, fmt.format(r.get_height()), ha="center", fontsize=size)


def status_by_label(df: pd.DataFrame, out: Path):
    ct = pd.crosstab(df["status_code"].astype(str), df["activity_label"]).reindex(columns=LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    bottom = np.zeros(len(ct))
    for k in LABELS:
        ax.bar(ct.index, ct[k], bottom=bottom, label=k, color=CLASS_COLORS[k], width=0.6)
        bottom += ct[k].values
    ax.set_yscale("log"); ax.set_ylim(1, 20000)
    for i, v in enumerate(bottom):
        ax.text(i, v * 1.15, f"{int(v):,}", ha="center", fontsize=8.5)
    ax.set_xlabel("HTTP status code"); ax.set_ylabel("Number of requests (log scale)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def split_distribution(y_train, y_test, out: Path):
    tr, te = y_train.value_counts(), y_test.value_counts()
    x, w = np.arange(3), 0.36
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    b1 = ax.bar(x - w / 2, [tr.get(k, 0) for k in LABELS], w, color=C_TEXT, label="Training (80%)")
    b2 = ax.bar(x + w / 2, [te.get(k, 0) for k in LABELS], w, color=C_CTX, label="Testing (20%)")
    _bar_labels(ax, b1, dy=60); _bar_labels(ax, b2, dy=60)
    ax.set_xticks(x, LABELS); ax.set_ylabel("Number of records"); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def overall_comparison(res: dict, out: Path):
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    names = ["Accuracy", "Macro precision", "Macro recall", "Macro-F1", "Weighted F1"]
    v1 = [res["Text-Based"][k] * 100 for k in keys]; v2 = [res["Context-Aware"][k] * 100 for k in keys]
    x, w = np.arange(len(keys)), 0.36
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    b1 = ax.bar(x - w / 2, v1, w, color=C_TEXT, label="Text-Based MNB")
    b2 = ax.bar(x + w / 2, v2, w, color=C_CTX, label="Context-Aware MNB")
    _bar_labels(ax, b1, "{:.2f}", 1, 7.5); _bar_labels(ax, b2, "{:.2f}", 1, 7.5)
    ax.set_xticks(x, names); ax.set_ylim(0, 112); ax.set_ylabel("Score (%)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def per_class_comparison(pc_text: pd.DataFrame, pc_ctx: pd.DataFrame, out: Path):
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
    w = 0.36
    for ax, col, title in zip(axs, ["precision", "recall", "f1"], ["Precision", "Recall", "F1-score"]):
        x = np.arange(3)
        r1 = ax.bar(x - w / 2, pc_text[col] * 100, w, color=C_TEXT, label="Text-Based")
        r2 = ax.bar(x + w / 2, pc_ctx[col] * 100, w, color=C_CTX, label="Context-Aware")
        _bar_labels(ax, r1, "{:.1f}", 1.5, 6.5); _bar_labels(ax, r2, "{:.1f}", 1.5, 6.5)
        ax.set_title(title, fontsize=10); ax.set_xticks(x, SHORT, fontsize=7.5); ax.set_ylim(0, 112)
    axs[0].set_ylabel("Score (%)")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out); plt.close(fig)


def confusion_pair(cm_text: np.ndarray, cm_ctx: np.ndarray, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for ax, cm, title in zip(axs, [cm_text, cm_ctx], ["(a) Text-Based MNB", "(b) Context-Aware MNB"]):
        nr = cm / cm.sum(1, keepdims=True)
        ax.imshow(nr, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:,}\n({nr[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=7.5,
                        color="white" if nr[i, j] > 0.55 else "black")
        ax.set_xticks(range(3), SHORT, fontsize=7.5); ax.set_yticks(range(3), SHORT, fontsize=7.5)
        ax.set_xlabel("Predicted label", fontsize=8.5); ax.set_title(title, fontsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("Actual label", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def robustness(rob: pd.DataFrame, out: Path):
    """rob: robustness_summary table (scheme, config, <metric>_mean, <metric>_sd)."""
    schemes = ["Stratified 5-fold CV", "IP-grouped 5-fold CV", "Temporal 80/20 split"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    x, w = np.arange(3), 0.36
    for ax, metric, title in zip(axs, ["macro_f1", "suspicious_f1"], ["Macro-F1", "Suspicious F1-score"]):
        for off, cfg, col in [(-w / 2, "Text-Based", C_TEXT), (w / 2, "Context-Aware", C_CTX)]:
            sub = rob[rob["config"] == cfg].set_index("scheme").reindex(schemes)
            bars = ax.bar(x + off, sub[f"{metric}_mean"] * 100, w, yerr=sub[f"{metric}_sd"] * 100, capsize=3,
                          color=col, label=f"{cfg} MNB", error_kw={"elinewidth": 0.8})
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, 2, f"{r.get_height():.1f}", ha="center", fontsize=7,
                        color="white" if cfg == "Context-Aware" else "black")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, ["Stratified\n5-fold", "IP-grouped\n5-fold", "Temporal\n80/20"], fontsize=8)
        ax.set_ylim(0, 125)
    axs[0].set_ylabel("Score (%)")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out); plt.close(fig)
