#!/usr/bin/env python3
"""
EPL Oracle v4.2 — ML Model Trainer
Gradient Boosting classifier (GBM) with isotonic calibration.
Exports decision trees to JSON so TypeScript can do inference without scipy.

Outputs:
  data/model/gbm_model.json     — serialized tree ensemble for TS inference
  data/model/scaler.json        — StandardScaler params
  data/model/calibration_*.json — per-class isotonic calibration
  data/model/metadata.json      — training metrics

Requirements:
  pip install scikit-learn numpy pandas

Usage: python python/train_model.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Install with: pip install scikit-learn numpy pandas")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent
DATA_PATH  = ROOT_DIR / "data" / "training_data.csv"
MODEL_DIR  = ROOT_DIR / "data" / "model"

# Must match fetch_historical.py FEATURE_NAMES exactly
FEATURE_NAMES = [
    "elo_diff", "attack_diff", "defense_diff", "goal_diff_diff",
    "goals_for_diff", "goals_against_diff", "net_goals_diff",
    "form_diff", "home_form", "away_form",
    "position_diff", "shots_on_target_diff", "possession_diff", "clean_sheet_diff",
    "rest_days_diff", "home_short_rest", "away_short_rest",
    "home_euro_fatigue", "away_euro_fatigue", "is_neutral",
    "lambda_home", "lambda_away",
    "vegas_home_prob", "vegas_draw_prob", "mc_home_win_prob",
    # v4.2: home/away split
    "home_att_home", "home_def_home", "away_att_away", "away_def_away",
    # v4.2: head-to-head
    "h2h_home_win_rate", "h2h_goal_diff",
]


def serialize_tree(estimator) -> dict:
    """Serialize a sklearn DecisionTreeRegressor to a compact JSON-friendly dict."""
    tree = estimator.tree_
    return {
        "left":      tree.children_left.tolist(),
        "right":     tree.children_right.tolist(),
        "feature":   tree.feature.tolist(),
        "threshold": [round(float(t), 8) for t in tree.threshold],
        # value shape: (n_nodes, 1, 1) for single-output regression tree
        "value":     [round(float(tree.value[i][0][0]), 8) for i in range(tree.node_count)],
    }


def serialize_gbm(model, scaler, classes: list, X_scaled_sample) -> dict:
    """
    Serialize a fitted GradientBoostingClassifier for TypeScript inference.

    Inference formula (TypeScript):
      scaled_x = (x - scaler.mean) / scaler.scale
      raw[k] = init_scores[k]
      for each stage:
          raw[k] += learning_rate * tree[stage][k].predict(scaled_x)
      probs = softmax(raw)
    """
    # Compute init scores (constant across all inputs for DummyClassifier prior)
    init_scores = model._raw_predict_init(X_scaled_sample[:1])[0].tolist()

    # Serialize all trees: shape (n_estimators, n_classes)
    trees = []
    for stage in model.estimators_:       # stage: array of (n_classes,) estimators
        stage_trees = [serialize_tree(est) for est in stage]
        trees.append(stage_trees)

    return {
        "version":       "4.2.0",
        "n_features":    len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "classes":       classes,
        "n_classes":     len(classes),
        "n_estimators":  len(trees),
        "learning_rate": model.learning_rate,
        "init_scores":   [round(s, 8) for s in init_scores],
        "trees":         trees,  # [n_estimators][n_classes] each has {left,right,feature,threshold,value}
    }


def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Training data not found at {DATA_PATH}")
        print("       Run 'python python/fetch_historical.py' first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} training samples")

    for cls in ["home", "draw", "away"]:
        n = (df["actual_outcome"] == cls).sum()
        print(f"  {cls}: {n} ({n/len(df)*100:.1f}%)")

    if len(df) < 200:
        print(f"[WARNING] Only {len(df)} samples — model may underfit.")

    X = df[FEATURE_NAMES].fillna(0.0).values
    y = df["actual_outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_all_s   = scaler.transform(X)

    # ── Train Gradient Boosting (captures non-linear feature interactions) ─────
    print("\n[INFO] Training GradientBoostingClassifier ...")
    model = GradientBoostingClassifier(
        n_estimators=200,       # 200 boosting rounds
        max_depth=3,            # shallow trees prevent overfitting
        learning_rate=0.05,     # conservative step size (pairs with high n_estimators)
        subsample=0.8,          # stochastic GBM reduces overfitting
        min_samples_leaf=10,    # minimum leaf size
        max_features="sqrt",    # feature subsampling per split
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_all_s, y, cv=cv, scoring="accuracy")
    print(f"\n[CV] 5-fold accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    classes = list(model.classes_)

    test_acc = accuracy_score(y_test, y_pred)
    test_ll  = log_loss(y_test, y_prob, labels=classes)
    print(f"[TEST] Accuracy: {test_acc:.3f}  |  Log-Loss: {test_ll:.4f}")

    brier_scores = []
    for i, cls in enumerate(classes):
        y_bin = (y_test == cls).astype(float)
        bs    = brier_score_loss(y_bin, y_prob[:, i])
        brier_scores.append(bs)
        print(f"  Brier ({cls}): {bs:.4f}")
    avg_brier = float(np.mean(brier_scores))
    print(f"  Avg Brier: {avg_brier:.4f}")

    # ── Isotonic calibration per class (fit on full dataset) ─────────────────
    y_prob_all = model.predict_proba(X_all_s)
    calibrators = {}
    for i, cls in enumerate(classes):
        y_bin = (y == cls).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(y_prob_all[:, i], y_bin)
        calibrators[cls] = ir
        print(f"[CAL] Isotonic calibration fitted for class '{cls}'")

    # ── Save artifacts ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # GBM model as serialized trees for TypeScript inference
    gbm_data = serialize_gbm(model, scaler, classes, X_all_s)
    with open(MODEL_DIR / "gbm_model.json", "w") as f:
        json.dump(gbm_data, f, separators=(",", ":"))  # compact JSON
    print(f"[OK] GBM saved ({len(gbm_data['trees'])} stages x {gbm_data['n_classes']} trees)")

    # Scaler params (for standardization before tree traversal)
    scaler_data = {
        "mean":          scaler.mean_.tolist(),
        "scale":         scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(MODEL_DIR / "scaler.json", "w") as f:
        json.dump(scaler_data, f, indent=2)

    # Per-class isotonic calibration
    for cls, ir in calibrators.items():
        cal_data = {
            "x_thresholds": ir.X_thresholds_.tolist(),
            "y_thresholds": ir.y_thresholds_.tolist(),
        }
        with open(MODEL_DIR / f"calibration_{cls}.json", "w") as f:
            json.dump(cal_data, f, indent=2)

    # Metadata
    seasons = sorted(df["season"].unique().tolist())
    meta = {
        "version":           "4.2.0",
        "model_type":        "GradientBoosting",
        "train_seasons":     f"{seasons[0]}-{seasons[-1]}",
        "avg_brier":         avg_brier,
        "avg_accuracy":      float(test_acc),
        "cv_accuracy_mean":  float(cv_scores.mean()),
        "cv_accuracy_std":   float(cv_scores.std()),
        "n_features":        len(FEATURE_NAMES),
        "n_train_samples":   len(X_train),
        "n_test_samples":    len(X_test),
        "n_total_samples":   len(X),
        "trained_at":        datetime.now().isoformat(),
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Model saved to {MODEL_DIR}")
    print(f"     Accuracy: {test_acc:.3f}  |  Avg Brier: {avg_brier:.4f}")
    print(f"     GBM model size: {(MODEL_DIR / 'gbm_model.json').stat().st_size / 1024:.0f} KB")

    # ── Feature importance ────────────────────────────────────────────────────
    print("\nTop features by GBM importance:")
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    for name, imp in ranked[:12]:
        bar = "#" * int(imp * 200)
        print(f"  {name:<32} {imp:.4f}  {bar}")


if __name__ == "__main__":
    train()
