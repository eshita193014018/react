
import pandas as pd
from dataset_loader import load_dataset
from preprocessing import tokenize_and_lemmatize
from feature_engineering import fit_vectorizer, save_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from model_utils import save_model, print_metrics

df = load_dataset("dataset.csv")
if not {"text", "label"}.issubset(df.columns):
    raise ValueError("dataset.csv must have 'text' and 'label' columns.")

df["clean"] = df["text"].astype(str).apply(tokenize_and_lemmatize)

X = df["clean"]
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Vectorizer on train only
vec = fit_vectorizer(X_train)
save_vectorizer(vec)
X_train_vec = vec.transform(X_train)
X_test_vec  = vec.transform(X_test)

# Use a solver that supports sparse input
model = LogisticRegression(max_iter=1000, solver="liblinear")  # or solver="saga"
model.fit(X_train_vec, y_train)
save_model(model)

preds = model.predict(X_test_vec)
print_metrics(y_test, preds)
