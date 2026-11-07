"""Evaluation helpers: metrics, McNemar test, cross-validation schemes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .labeling import LABELS


def overall_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "suspicious_f1": f1_score(y_true, y_pred, labels=["Suspicious"], average="macro", zero_division=0),
        "suspicious_recall": recall_score(y_true, y_pred, labels=["Suspicious"], average="macro", zero_division=0),
    }


def per_class(y_true, y_pred) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tp = np.diag(cm)
    return pd.DataFrame({"class": LABELS, "precision": p, "recall": r, "f1": f, "support": s,
                         "predicted": cm.sum(0), "TP": tp, "FP": cm.sum(0) - tp, "FN": cm.sum(1) - tp})


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    y = np.asarray(y_true)
    a = np.asarray(pred_a) == y
    b = np.asarray(pred_b) == y
    n01 = int(np.sum(a & ~b))  # only A correct
    n10 = int(np.sum(~a & b))  # only B correct
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"only_a_correct": n01, "only_b_correct": n10, "p_value": float(min(1.0, p))}


def cross_validate(model, X: pd.DataFrame, y: pd.Series, groups=None, n_splits=5, seed=42) -> pd.DataFrame:
    """Stratified (or stratified-group) k-fold; returns one row of metrics per fold."""
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y, groups)
    rows = []
    for k, (tr, te) in enumerate(splits, start=1):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        row = overall_metrics(y.iloc[te], pred)
        row.update({"fold": k, "n_test": len(te), "n_test_suspicious": int((y.iloc[te] == "Suspicious").sum())})
        rows.append(row)
    return pd.DataFrame(rows)
