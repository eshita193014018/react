# feature_engineering.py
import os
import pickle
from typing import Iterable, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

VECT_PATH = "vectorizer.pkl"

def fit_vectorizer(
    corpus: Iterable[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    corpus = list(corpus)
    if not corpus:
        raise ValueError("corpus is empty; provide at least one document.")
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vec.fit(corpus)
    return vec

def save_vectorizer(vec: TfidfVectorizer, path: str = VECT_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(vec, f)

def load_vectorizer(path: str = VECT_PATH) -> TfidfVectorizer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Train vectorizer first.")
    with open(path, "rb") as f:
        return pickle.load(f)