#!/usr/bin/env python3
"""
EPL Oracle v4.2 -- Walk-Forward Backtest
Loads training_data.csv and runs walk-forward cross-validation using
GradientBoostingClassifier (3-way: home/draw/away).
Reports per-season accuracy, Brier score, and feature importance.

Usage:
  python python/fetch_historical.py  # if training_data.csv does not exist yet
  python python/backtest.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH  = PROJECT_ROOT / "data" / "training_data.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "model"

FEATURE_NAMES = [
    "elo_diff", "attack_diff", "defense_diff", "goal_diff_diff",
    "goals_for_diff", "goals_against_diff", "net_goals_diff",
    "form_diff", "home_form", "away_form",
    "position_diff", "shots_on_target_diff", "possession_diff", "clean_sheet_diff",
    "rest_days_diff", "home_short_rest", "away_short_rest",
    "home_euro_fatigue", "away_euro_fatigue", "is_neutral",
    "lambda_home", "lambda_away",
    "vegas_home_prob", "vegas_draw_prob", "mc_home_win_prob",
    "home_att_home", "home_def_home", "away_att_away", "away_def_away",
    "h2h_home_win_rate", "h2h_goal_diff",
    "line_movement_home", "corners_diff", "referee_home_bias",
]
TARGET = "actual_outcome"  # "home", "draw", "away"
CLASSES = ["home", "draw", "away"]


# -- 3-way Brier score --------------------------------------------------------

def brier_3way(probs_list, outcomes):
    outcome_map = {"home": [1,0,0], "draw": [0,1,0], "away": [0,0,1]}
    scores = []
    for probs, outcome in zip(probs_list, outcomes):
        actual = outcome_map.get(outcome, [0,0,0])
        scores.append(sum((probs[j] - actual[j])**2 for j in range(3)))
    return sum(scores) / len(scores) if scores else 0.0


def accuracy_3way(preds, outcomes):
    pred_classes = [CLASSES[np.argmax(p)] for p in preds]
    return sum(p == o for p, o in zip(pred_classes, outcomes)) / len(outcomes)


def hc_accuracy_3way(preds, outcomes, threshold=0.55):
    pairs = [(p, o) for p, o in zip(preds, outcomes) if max(p) >= threshold]
    if not pairs:
        return None, 0
    correct = sum(CLASSES[np.argmax(p)] == o for p, o in pairs)
    return correct / len(pairs), len(pairs)


def simulate_roi_3way(preds, outcomes, min_conf=0.50, bet=10.0, bankroll=1000.0):
    w = l = 0
    bank = bankroll
    for probs, outcome in zip(preds, outcomes):
        pick_idx = int(np.argmax(probs))
        if probs[pick_idx] < min_conf:
            continue
        fair_odds = 1.0 / max(probs[pick_idx], 0.01)
        offered = fair_odds * 0.95
        if CLASSES[pick_idx] == outcome:
            bank += bet * (offered - 1)
            w += 1
        else:
            bank -= bet
            l += 1
    roi = (bank - bankroll) / ((w + l) * bet) * 100 if (w + l) > 0 else 0.0
    return roi, w, l


# -- Walk-forward CV ----------------------------------------------------------

def walk_forward(df):
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        print("Need at least 2 seasons for walk-forward CV.")
        return {}, [], []

    all_preds, all_outcomes = [], []
    season_results = {}

    print("Walk-forward results:")
    print(f"  {'Season':>8}  {'N':>5}  {'Acc':>6}  {'Brier':>7}  {'HC%':>7}  {'HC N':>6}  {'ROI%':>7}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*7}")

    for i, test_season in enumerate(seasons[1:], 1):
        train_df = df[df["season"].isin(seasons[:i])]
        test_df  = df[df["season"] == test_season]
        if len(train_df) < 60 or len(test_df) < 20:
            continue

        feats = [f for f in FEATURE_NAMES if f in train_df.columns]
        X_tr = train_df[feats].fillna(0).values
        y_tr = train_df[TARGET].values
        X_te = test_df[feats].fillna(0).values
        y_te = test_df[TARGET].values

        # Time-decay weights: recent seasons matter more
        current_max = seasons[i]  # last training season
        weights = train_df["season"].apply(lambda s: 0.85 ** (current_max - s)).values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X_tr_s, y_tr, sample_weight=weights)

        preds = model.predict_proba(X_te_s)  # shape (n, 3): columns = model.classes_
        # Reorder to [H, D, A]
        class_order = list(model.classes_)
        ordered_preds = np.zeros((len(preds), 3))
        for j, cls in enumerate(CLASSES):
            if cls in class_order:
                ordered_preds[:, j] = preds[:, class_order.index(cls)]

        acc   = accuracy_3way(ordered_preds, y_te)
        brier = brier_3way(ordered_preds, y_te)
        hc_acc, hc_n = hc_accuracy_3way(ordered_preds, y_te, threshold=0.55)
        roi, wins, losses = simulate_roi_3way(ordered_preds, y_te)

        season_results[str(test_season)] = {
            "n": len(test_df),
            "accuracy": round(float(acc), 4),
            "brier": round(float(brier), 4),
            "hc_accuracy": round(float(hc_acc), 4) if hc_acc is not None else None,
            "hc_n": int(hc_n),
            "roi_pct": round(roi, 2),
        }

        hc_str = f"{hc_acc:.3f}" if hc_acc is not None else "   N/A"
        print(f"  {str(test_season):>8}  {len(test_df):>5}  {acc:>6.3f}  {brier:>7.4f}  "
              f"{hc_str:>7}  {hc_n:>6}  {roi:>+7.1f}%")

        all_preds.extend(ordered_preds.tolist())
        all_outcomes.extend(y_te.tolist())

    return season_results, all_preds, all_outcomes


# -- Feature importance -------------------------------------------------------

def print_feature_importance(df):
    feats = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[feats].fillna(0).values
    y = df[TARGET].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                        learning_rate=0.05, random_state=42)
    model.fit(X_s, y)
    importances = model.feature_importances_
    ranked = sorted(zip(feats, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance (GBM feature importance, full dataset):")
    for name, val in ranked[:15]:
        bar = "#" * int(val * 200)
        print(f"  {name:<34} {val:>6.4f}  {bar}")


# -- Main ---------------------------------------------------------------------

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        print("  Run: python python/fetch_historical.py")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print("EPL Oracle v4.2 -- Walk-Forward Backtest")
    print("=" * 50)
    print(f"Loaded {len(df)} matches, seasons: {sorted(df['season'].unique().tolist())}")
    if TARGET in df.columns:
        dist = df[TARGET].value_counts()
        print(f"Outcome distribution: H={dist.get('H',0)} ({dist.get('H',0)/len(df):.1%}), "
              f"D={dist.get('D',0)} ({dist.get('D',0)/len(df):.1%}), "
              f"A={dist.get('A',0)} ({dist.get('A',0)/len(df):.1%})")
    print()

    season_results, all_preds, all_outcomes = walk_forward(df)

    if len(all_preds) == 0:
        print("No predictions generated.")
        return

    print()
    print("Aggregate (all out-of-sample seasons):")
    print(f"  Total matches:       {len(all_preds)}")
    print(f"  Accuracy:            {accuracy_3way(all_preds, all_outcomes):.4f}  (3-way baseline ~0.465)")
    print(f"  Brier score (3-way): {brier_3way(all_preds, all_outcomes):.4f}  (lower = better, naive ~0.63)")
    hc_acc, hc_n = hc_accuracy_3way(all_preds, all_outcomes, threshold=0.55)
    if hc_acc is not None:
        print(f"  HC accuracy (>=55%): {hc_acc:.4f}  n={hc_n}")
    roi, w, l = simulate_roi_3way(all_preds, all_outcomes)
    print(f"  Simulated ROI:       {roi:+.2f}%  (W={w}, L={l})")

    print_feature_importance(df)
    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
