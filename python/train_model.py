#!/usr/bin/env python3
"""
EPL Oracle v4.1 — ML Model Trainer
Multi-class logistic regression (softmax) with isotonic calibration.
Outputs: data/model/coefficients.json, scaler.json, calibration_*.json, metadata.json

Requirements:
  pip install scikit-learn numpy pandas

Usage: python python/train_model.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Install with: pip install scikit-learn numpy pandas")
    sys.exit(1)

# ─── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "training_data.csv"
MODEL_DIR = ROOT_DIR / "data" / "model"

FEATURE_NAMES = [
    "elo_diff", "attack_diff", "defense_diff", "goal_diff_diff",
    "goals_for_diff", "goals_against_diff", "net_goals_diff",
    "form_diff", "home_form", "away_form",
    "position_diff", "shots_on_target_diff", "possession_diff", "clean_sheet_diff",
    "rest_days_diff", "home_short_rest", "away_short_rest",
    "home_euro_fatigue", "away_euro_fatigue", "is_neutral",
    "lambda_home", "lambda_away",
    "vegas_home_prob", "vegas_draw_prob", "mc_home_win_prob",
]

def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Training data not found at {DATA_PATH}")
        print("       Run 'npm run build-dataset' first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} training samples")

    # Class distribution
    for cls in ["home", "draw", "away"]:
        n = (df["actual_outcome"] == cls).sum()
        print(f"  {cls}: {n} ({n/len(df)*100:.1f}%)")

    if len(df) < 100:
        print(f"[WARNING] Only {len(df)} samples — model may underfit. Collect more data.")

    X = df[FEATURE_NAMES].fillna(0.0).values
    y = df["actual_outcome"].values  # 'home', 'draw', 'away'

    # ── Train/test split (80/20, stratified) ──────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Standardize features ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train multinomial logistic regression ─────────────────────────────────
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=0.8,           # L2 regularization
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring="accuracy")
    print(f"\n[CV] 5-fold accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    classes = list(model.classes_)

    test_acc = accuracy_score(y_test, y_pred)
    test_ll = log_loss(y_test, y_prob, labels=classes)
    print(f"[TEST] Accuracy: {test_acc:.3f}  |  Log-Loss: {test_ll:.4f}")

    # Brier score per class (one-vs-rest)
    brier_scores = []
    for i, cls in enumerate(classes):
        y_bin = (y_test == cls).astype(float)
        bs = brier_score_loss(y_bin, y_prob[:, i])
        brier_scores.append(bs)
        print(f"  Brier ({cls}): {bs:.4f}")
    avg_brier = np.mean(brier_scores)
    print(f"  Avg Brier: {avg_brier:.4f}")

    # ── Isotonic calibration per class ────────────────────────────────────────
    # Re-fit on full dataset for final calibration
    X_all_scaled = scaler.transform(X)
    y_prob_all = model.predict_proba(X_all_scaled)

    calibrators = {}
    for i, cls in enumerate(classes):
        y_bin = (y == cls).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_prob_all[:, i], y_bin)
        calibrators[cls] = ir
        print(f"[CAL] Isotonic calibration fitted for class '{cls}'")

    # ── Save artifacts ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # classes order matches model.classes_
    coef_data = {
        "classes": list(classes),
        "intercepts": model.intercept_.tolist(),
        "coefficients": model.coef_.tolist(),  # shape: (n_classes, n_features)
        "feature_names": FEATURE_NAMES,
    }
    with open(MODEL_DIR / "coefficients.json", "w") as f:
        json.dump(coef_data, f, indent=2)

    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(MODEL_DIR / "scaler.json", "w") as f:
        json.dump(scaler_data, f, indent=2)

    for cls, ir in calibrators.items():
        cal_data = {
            "x_thresholds": ir.X_thresholds_.tolist(),
            "y_thresholds": ir.y_thresholds_.tolist(),
        }
        with open(MODEL_DIR / f"calibration_{cls}.json", "w") as f:
            json.dump(cal_data, f, indent=2)

    seasons = sorted(df["season"].unique().tolist())
    meta = {
        "version": "4.1.0",
        "train_seasons": f"{seasons[0]}-{seasons[-1]}",
        "avg_brier": float(avg_brier),
        "avg_accuracy": float(test_acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_features": len(FEATURE_NAMES),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "trained_at": datetime.now().isoformat(),
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Model saved to {MODEL_DIR}")
    print(f"     Accuracy: {test_acc:.3f}  |  Avg Brier: {avg_brier:.4f}")
    print(f"     Restart the pipeline: npm start")

    # ── Feature importance ─────────────────────────────────────────────────────
    print("\nTop features by coefficient magnitude (avg across classes):")
    coef_abs = np.abs(model.coef_).mean(axis=0)
    ranked = sorted(zip(FEATURE_NAMES, coef_abs), key=lambda x: x[1], reverse=True)
    for name, importance in ranked[:10]:
        print(f"  {name:<30} {importance:.4f}")


if __name__ == "__main__":
    train()
