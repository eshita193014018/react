# evaluate.py
from __future__ import annotations

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import load_vectorizer
from model_utils import load_model, print_metrics, plot_confusion
from preprocessing import tokenize_and_lemmatize
from dataset_loader import load_dataset


def evaluate_on(
    path: str = "dataset.csv",
    test_size_fraction: float = 0.2,
    random_state: int = 42,
    confusion_out: str = "confusion_test.png",
) -> None:
    df = load_dataset(path)

    # Basic schema check (same columns as before)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must have 'text' and 'label' columns.")

    # Stratify only if labels have more than one class
    y = df["label"]
    stratify = y if y.nunique(dropna=False) > 1 else None

    # Deterministic split (same idea as old code)
    _, df_test = train_test_split(
        df,
        test_size=test_size_fraction,
        random_state=random_state,
        stratify=stratify,
    )
    df_test = df_test.copy()

    # Load artifacts
    vec = load_vectorizer()
    model = load_model()

    # Preprocess and vectorize (same as old, just safer assignment)
    df_test["clean"] = df_test["text"].astype(str).apply(tokenize_and_lemmatize)
    X_test_vec = vec.transform(df_test["clean"])
    y_test = df_test["label"].astype(int)

    # Predict & report
    preds = model.predict(X_test_vec)
    print_metrics(y_test, preds)

    # Save confusion matrix
    if confusion_out:
        plot_confusion(y_test, preds, out_path=confusion_out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    evaluate_on(path)
