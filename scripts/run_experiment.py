#!/usr/bin/env python
"""End-to-end experiment: parsing -> labeling -> splitting -> modelling -> evaluation.

Usage:
    python scripts/run_experiment.py --log data/unh.ac.id-ssl_log.txt --out outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from logcls import plots  # noqa: E402
from logcls.evaluation import cross_validate, mcnemar_exact, overall_metrics, per_class  # noqa: E402
from logcls.labeling import LABELS, apply_labels, rules_table  # noqa: E402
from logcls.models import CONFIGS, add_text_feature, make_model  # noqa: E402
from logcls.parsing import parse_log  # noqa: E402

SEED = 42


def anonymize_ip(ip: str, salt: str = "unh-log-study") -> str:
    return "ip_" + hashlib.sha256((salt + str(ip)).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to Apache/NGINX combined access log")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--font", default=None, help="Optional font family for figures")
    ap.add_argument("--keep-ip", action="store_true", help="Do not anonymise IP addresses in exported CSV files")
    args = ap.parse_args()

    out = Path(args.out); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    plots.setup(args.font)
    T = lambda name: out / "tables" / name  # noqa: E731
    Fg = lambda name: out / "figures" / name  # noqa: E731

    # 1. Parsing -----------------------------------------------------------
    df, failed, summary = parse_log(args.log)
    failed.to_csv(T("failed_lines.csv"), index=False)

    # 2. Labeling ----------------------------------------------------------
    df = apply_labels(df)
    rules_table().to_csv(T("labeling_rules.csv"), index=False)
    df.groupby(["activity_label", "label_rule"]).size().rename("records").to_csv(T("label_rule_counts.csv"))
    pd.crosstab(df["activity_label"], df["status_code"], margins=True, margins_name="Total").to_csv(T("label_by_status.csv"))
    pd.crosstab(df["activity_label"], df["http_method"], margins=True, margins_name="Total").to_csv(T("label_by_method.csv"))
    summary["review_records"] = int((df["activity_label"] == "Review").sum())
    summary["unique_ip_suspicious"] = int(df.loc[df["activity_label"] == "Suspicious", "ip"].nunique())
    summary["http_method_counts"] = df["http_method"].value_counts().to_dict()
    summary["protocol_counts"] = df["protocol"].value_counts().to_dict()
    summary["status_counts"] = df["status_code"].value_counts().to_dict()

    data = add_text_feature(df[df["activity_label"] != "Review"].reset_index(drop=True))
    X, y = data, data["activity_label"]
    dist = y.value_counts().reindex(LABELS)

    # 3. Stratified hold-out split ------------------------------------------
    idx_tr, idx_te = train_test_split(np.arange(len(data)), test_size=0.20, random_state=SEED, stratify=y)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
    pd.DataFrame({"records": dist, "share_pct": dist / dist.sum() * 100,
                  "train": ytr.value_counts().reindex(LABELS), "test": yte.value_counts().reindex(LABELS)}
                 ).to_csv(T("label_distribution.csv"))

    # 4. Hold-out evaluation: MNB with four feature configurations (ablation) -
    preds, holdout, dims = {}, {}, {}
    for cfg in CONFIGS:
        m = make_model(cfg, "MNB").fit(Xtr, ytr)
        preds[cfg] = m.predict(Xte)
        holdout[cfg] = overall_metrics(yte, preds[cfg])
        dims[cfg] = int(m.named_steps["features"].transform(Xtr.iloc[:1]).shape[1])
    abl = pd.DataFrame(holdout).T; abl["n_features"] = pd.Series(dims)
    abl.to_csv(T("holdout_ablation_mnb.csv"))

    pc_text, pc_ctx = per_class(yte, preds["Text-Based"]), per_class(yte, preds["Context-Aware"])
    pc_text.to_csv(T("per_class_text_based.csv"), index=False); pc_ctx.to_csv(T("per_class_context_aware.csv"), index=False)
    cm_text = confusion_matrix(yte, preds["Text-Based"], labels=LABELS)
    cm_ctx = confusion_matrix(yte, preds["Context-Aware"], labels=LABELS)
    pd.DataFrame(cm_text, index=LABELS, columns=LABELS).to_csv(T("confusion_text_based.csv"))
    pd.DataFrame(cm_ctx, index=LABELS, columns=LABELS).to_csv(T("confusion_context_aware.csv"))
    mcn = mcnemar_exact(yte, preds["Text-Based"], preds["Context-Aware"])

    # 5. Reference classifiers on the same split ---------------------------
    ref = {}
    for cfg in ["Text-Based", "Context-Aware"]:
        for clf in ["MNB", "CNB", "LR"]:
            p = make_model(cfg, clf).fit(Xtr, ytr).predict(Xte)
            ref[f"{clf} | {cfg}"] = overall_metrics(yte, p)
    pd.DataFrame(ref).T.to_csv(T("holdout_reference_classifiers.csv"))

    # 6. Robustness: stratified 5-fold, IP-grouped 5-fold, temporal split ----
    rob_rows, fold_tables = [], []
    for scheme, groups in [("Stratified 5-fold CV", None), ("IP-grouped 5-fold CV", X["ip"])]:
        for cfg in ["Text-Based", "Context-Aware"]:
            cv = cross_validate(make_model(cfg, "MNB"), X, y, groups=groups, n_splits=5, seed=SEED)
            cv.insert(0, "config", cfg); cv.insert(0, "scheme", scheme); fold_tables.append(cv)
    folds = pd.concat(fold_tables, ignore_index=True)
    folds.to_csv(T("cv_folds.csv"), index=False)

    order = X["datetime"].argsort(kind="stable").to_numpy()
    cut = int(len(order) * 0.8)
    t_tr, t_te = order[:cut], order[cut:]
    temporal = {}
    for cfg in ["Text-Based", "Context-Aware"]:
        p = make_model(cfg, "MNB").fit(X.iloc[t_tr], y.iloc[t_tr]).predict(X.iloc[t_te])
        temporal[cfg] = overall_metrics(y.iloc[t_te], p)
    temporal_info = {"train_end": str(X["datetime"].iloc[t_tr].max()), "test_start": str(X["datetime"].iloc[t_te].min()),
                     "test_suspicious": int((y.iloc[t_te] == "Suspicious").sum()),
                     "train_suspicious": int((y.iloc[t_tr] == "Suspicious").sum())}

    for scheme in ["Stratified 5-fold CV", "IP-grouped 5-fold CV"]:
        f = folds[folds["scheme"] == scheme]
        for cfg in ["Text-Based", "Context-Aware"]:
            g = f[f["config"] == cfg]
            rob_rows.append({"scheme": scheme, "config": cfg,
                             **{f"{k}_mean": g[k].mean() for k in ["accuracy", "macro_f1", "suspicious_f1", "suspicious_recall"]},
                             **{f"{k}_sd": g[k].std(ddof=1) for k in ["accuracy", "macro_f1", "suspicious_f1", "suspicious_recall"]}})
    for cfg in ["Text-Based", "Context-Aware"]:
        rob_rows.append({"scheme": "Temporal 80/20 split", "config": cfg,
                         **{f"{k}_mean": temporal[cfg][k] for k in ["accuracy", "macro_f1", "suspicious_f1", "suspicious_recall"]},
                         **{f"{k}_sd": 0.0 for k in ["accuracy", "macro_f1", "suspicious_f1", "suspicious_recall"]}})
    rob = pd.DataFrame(rob_rows); rob.to_csv(T("robustness_summary.csv"), index=False)

    # 7. Predictions and error analysis ------------------------------------
    pred_df = Xte[["ip", "datetime", "http_method", "url", "status_code", "user_agent", "label_rule"]].copy()
    pred_df["actual"] = yte.values
    for cfg in CONFIGS:
        pred_df[f"pred_{cfg}"] = preds[cfg]
    if not args.keep_ip:
        pred_df["ip"] = pred_df["ip"].map(anonymize_ip)
    pred_df.to_csv(T("holdout_predictions.csv"), index=False)
    fn = pred_df[(pred_df["actual"] == "Suspicious") & (pred_df["pred_Context-Aware"] != "Suspicious")]
    fn.to_csv(T("context_aware_suspicious_false_negatives.csv"), index=False)

    # 8. Figures -----------------------------------------------------------
    plots.status_by_label(df[df["activity_label"] != "Review"], Fg("fig_status_by_label.png"))
    plots.split_distribution(ytr, yte, Fg("fig_split_distribution.png"))
    plots.overall_comparison(holdout, Fg("fig_overall_comparison.png"))
    plots.per_class_comparison(pc_text, pc_ctx, Fg("fig_per_class.png"))
    plots.confusion_pair(cm_text, cm_ctx, Fg("fig_confusion_matrices.png"))
    plots.robustness(rob, Fg("fig_robustness.png"))

    # 9. Export labelled dataset (anonymised) and metrics -------------------
    export = df.copy()
    if not args.keep_ip:
        export["ip"] = export["ip"].map(anonymize_ip)
    export.to_csv(out / "http_log_labeled.csv", index=False)
    results = {"summary": summary, "label_distribution": dist.to_dict(),
               "split": {"train": len(idx_tr), "test": len(idx_te)},
               "holdout": holdout, "n_features": dims, "mcnemar_text_vs_context": mcn,
               "temporal_split": temporal_info, "suspicious_false_negatives_context": len(fn)}
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({"holdout_macro_f1": {k: round(v["macro_f1"], 4) for k, v in holdout.items()}, "mcnemar": mcn}, indent=2))


if __name__ == "__main__":
    main()
