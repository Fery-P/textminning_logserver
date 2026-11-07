"""Feature pipelines and classifiers.

All transformations are wrapped in scikit-learn Pipelines so that the TF-IDF
vocabulary and the one-hot encoder are fitted on training folds only.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TFIDF_PARAMS = dict(
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b",  # scikit-learn default, stated explicitly
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
)

# Feature configurations used in the experiments (ablation included).
CONFIGS = {
    "Text-Based": [],
    "Text + Method": ["http_method"],
    "Text + Status": ["status_code"],
    "Context-Aware": ["http_method", "status_code"],
}


def add_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_feature"] = out["url"].fillna("").astype(str) + " " + out["user_agent"].fillna("").astype(str)
    out["http_method"] = out["http_method"].fillna("UNKNOWN").astype(str)
    out["status_code"] = out["status_code"].fillna("UNKNOWN").astype(str)
    return out


def make_features(context_cols: list[str]) -> ColumnTransformer:
    transformers = [("tfidf", TfidfVectorizer(**TFIDF_PARAMS), "text_feature")]
    if context_cols:
        transformers.append(("context", OneHotEncoder(handle_unknown="ignore"), context_cols))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def make_model(config: str = "Context-Aware", classifier: str = "MNB") -> Pipeline:
    if classifier == "MNB":
        clf = MultinomialNB(alpha=1.0)
    elif classifier == "CNB":
        clf = ComplementNB(alpha=1.0)
    elif classifier == "LR":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        raise ValueError(classifier)
    return Pipeline([("features", make_features(CONFIGS[config])), ("clf", clf)])
