#  update_app.py
from flask import Flask, request, render_template
from feature_engineering import load_vectorizer
from model_utils import load_model
from preprocessing import tokenize_and_lemmatize
import os
import numpy as np

app = Flask(name)

MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

# Load on startup (will raise if files are missing)
vec = load_vectorizer(VECT_PATH)
model = load_model(MODEL_PATH)

def predict_text(text: str):
    clean = tokenize_and_lemmatize(text)
    X = vec.transform([clean])
    pred = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        # get probability for class 1 if present; otherwise fall back to max
        probs = model.predict_proba(X)[0]  # shape: (n_classes,)
        if hasattr(model, "classes_") and 1 in getattr(model, "classes_", []):
            idx = list(model.classes_).index(1)
            proba = float(probs[idx])
        else:
            proba = float(np.max(probs))
    return int(pred), proba

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text.")
    pred, proba = predict_text(text)
    label = "Positive" if pred == 1 else "Negative"
    return render_template("index.html", prediction=label, probability=proba, text=text)

if name == "main":
    port = int(os.environ.get("PORT", 5000))
    # For production, set debug=False and host="0.0.0.0" on a real server (Gunicorn, etc.)
    app.run(debug=True, port=port)