
# dataset_loader.py
import pandas as pd
from typing import Tuple

POSSIBLE_TEXT_COLS = ["text", "content", "tweet", "message", "review"]
POSSIBLE_LABEL_COLS = ["label", "sentiment", "target", "score"]

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # find text col
    text_col = next((c for c in df.columns if c.lower() in POSSIBLE_TEXT_COLS), None)
    label_col = next((c for c in df.columns if c.lower() in POSSIBLE_LABEL_COLS), None)

    if text_col is None or label_col is None:
        # fallback: try heuristics
        for c in df.columns:
            if df[c].dtype == object and text_col is None:
                text_col = c
            if df[c].nunique() <= 10 and label_col is None and df[c].dtype in ['int64','float64','object']:
                label_col = c

    if text_col is None or label_col is None:
        raise ValueError("Could not auto-detect text or label column. Please supply a CSV with columns like 'text' and 'label'.")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    # normalize labels to 0/1 if possible
    unique = sorted(df['label'].dropna().unique())
    if set(unique) <= {"positive","negative","Neutral","neutral","pos","neg"}:
        df['label'] = df['label'].map(lambda x: 1 if str(x).lower().startswith('p') else 0)
    else:
        # if they are numeric but not 0/1, map first class to 0 and second to 1
        vals = list(pd.Series(df['label'].unique()).dropna())
        if len(vals) == 2 and set(vals) != {0,1}:
            mapping = {vals[0]: 0, vals[1]: 1}
            df['label'] = df['label'].map(mapping)
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
    return df

if name == "main":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    df = load_dataset(path)
    print("Loaded dataset with shape:", df.shape)
    print("Label distribution:\n", df['label'].value_counts(normalize= True ) )