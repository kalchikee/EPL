#!/usr/bin/env python3
"""
EPL Oracle v4.1 — Training Dataset Builder
Reads the SQLite database and exports a CSV for ML training.
Run after accumulating a full season (or more) of predictions + results.

Usage: python python/build_dataset.py
Output: data/training_data.csv
"""

import json
import os
import sqlite3
import csv
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DB_PATH = ROOT_DIR / "data" / "epl_oracle.db"
OUTPUT_PATH = ROOT_DIR / "data" / "training_data.csv"

# ─── Feature names (must match src/models/metaModel.ts FEATURE_NAMES) ─────────

FEATURE_NAMES = [
    "elo_diff",
    "attack_diff",
    "defense_diff",
    "goal_diff_diff",
    "goals_for_diff",
    "goals_against_diff",
    "net_goals_diff",
    "form_diff",
    "home_form",
    "away_form",
    "position_diff",
    "shots_on_target_diff",
    "possession_diff",
    "clean_sheet_diff",
    "rest_days_diff",
    "home_short_rest",
    "away_short_rest",
    "home_euro_fatigue",
    "away_euro_fatigue",
    "is_neutral",
    "lambda_home",
    "lambda_away",
    "vegas_home_prob",
    "vegas_draw_prob",
    "mc_home_win_prob",
]

def build_dataset():
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found at {DB_PATH}")
        print("       Run the pipeline first to generate predictions and results.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Only include rows where we have both a prediction and an outcome
    cursor.execute("""
        SELECT *
        FROM predictions
        WHERE actual_outcome IS NOT NULL
        AND correct IS NOT NULL
        ORDER BY season ASC, gameweek ASC
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[WARNING] No evaluated predictions found in the database.")
        print("          Run predictions for completed gameweeks and process results first.")
        return

    print(f"[INFO] Found {len(rows)} evaluated predictions")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = (
            FEATURE_NAMES
            + ["match_id", "gameweek", "season", "home_team", "away_team", "match_date"]
            + ["actual_outcome", "home_win_label", "draw_label", "away_win_label"]
        )
        writer.writerow(header)

        skipped = 0
        for row in rows:
            try:
                fv = json.loads(row["feature_vector"])
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue

            feature_vals = [fv.get(name, 0.0) for name in FEATURE_NAMES]

            actual = row["actual_outcome"]
            home_label = 1 if actual == "home" else 0
            draw_label = 1 if actual == "draw" else 0
            away_label = 1 if actual == "away" else 0

            meta = [
                row["match_id"],
                row["gameweek"],
                row["season"],
                row["home_team"],
                row["away_team"],
                row["match_date"],
            ]

            labels = [actual, home_label, draw_label, away_label]
            writer.writerow(feature_vals + meta + labels)

    total = len(rows) - skipped
    print(f"[INFO] Exported {total} rows to {OUTPUT_PATH}")
    if skipped:
        print(f"[WARNING] Skipped {skipped} rows with invalid feature vectors")

    # Season breakdown
    print("\nSeason breakdown:")
    from collections import Counter
    seasons = [row["season"] for row in rows]
    outcomes = [row["actual_outcome"] for row in rows]
    for season, count in sorted(Counter(seasons).items()):
        season_rows = [r for r in rows if r["season"] == season]
        home_pct = sum(1 for r in season_rows if r["actual_outcome"] == "home") / len(season_rows) * 100
        draw_pct = sum(1 for r in season_rows if r["actual_outcome"] == "draw") / len(season_rows) * 100
        away_pct = sum(1 for r in season_rows if r["actual_outcome"] == "away") / len(season_rows) * 100
        print(f"  {season}-{str(season+1)[2:]}: {count} matches  "
              f"H:{home_pct:.0f}% D:{draw_pct:.0f}% A:{away_pct:.0f}%")

    print(f"\n[OK] Dataset ready at: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()
