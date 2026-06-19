#!/usr/bin/env python3
"""EPL recap — grade past predictions against ESPN final scores.

EPL Oracle previously relied on a SQLite DB column (predictions.correct)
set by a separate recap path, but the DB lives in git and is never
committed back from CI, so `correct` always stayed NULL and the Discord
embed's season-accuracy field was hidden.

This script mirrors the WNBA Oracle pattern: walk every predictions/<date>.json
file, fetch ESPN's EPL scoreboard for that date and a small window after
(EPL matches scheduled Saturday morning UTC sometimes play Sunday), match
each pick to its final score, and append a graded row to
data/grading_history.json:

    {
      "graded": [{
        "date": "2026-05-23", "gameId": "epl-...-AVL-FUL",
        "home": "FUL", "away": "AVL", "pickedTeam": "AVL",
        "modelProb": 0.7261, "actualWinner": "FUL",   # or "DRAW"
        "correct": false, "homeScore": 2, "awayScore": 1,
        "gradedAt": "..."
      }]
    }

Picks are graded as wrong if the actual outcome is a draw (the JSON
pickedSide is binary; ties go against the bot). Re-running is idempotent.

Usage:
    python python/recap.py                   # grade everything ungraded
    python python/recap.py --date 2026-05-23 # grade just one date
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
PREDICTIONS_DIR = ROOT / "predictions"
HISTORY_FILE = ROOT / "data" / "grading_history.json"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"

# Maps prediction-side abbreviation -> ESPN abbreviation. EPL Oracle and
# ESPN agree on most 3-letter codes; this table catches the few they don't.
ABBR_ALIASES: dict[str, str] = {
    "MUN": "MAN",   # EPL Oracle uses MUN for Man United; ESPN uses MAN
    "MCI": "MNC",   # Man City — verify if either form appears
}

# How many days after the prediction date to keep searching ESPN for the
# match. EPL fixtures are typically scheduled for the day of the prediction
# alert, but rescheduled matches can land 1-2 days later.
LOOKAHEAD_DAYS = 3


def normalize_abbr(s: str) -> str:
    return ABBR_ALIASES.get(s, s)


def load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {"graded": []}
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception as e:
        backup = HISTORY_FILE.with_suffix(
            f".corrupt-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
        )
        try:
            HISTORY_FILE.rename(backup)
            print(f"[recap] CORRUPT history file — preserved as {backup}; "
                  f"starting fresh: {e}")
        except Exception:
            print(f"[recap] CORRUPT history file — could not back up; "
                  f"starting fresh: {e}")
        return {"graded": []}


def save_history(history: dict) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, indent=2))
    tmp.replace(HISTORY_FILE)


def fetch_espn_scoreboard(iso_date: str) -> list[dict]:
    """Return completed games for the given date as a list of dicts."""
    yyyymmdd = iso_date.replace("-", "")
    url = f"{ESPN_BASE}?dates={yyyymmdd}&limit=50"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[recap] ESPN fetch failed for {iso_date}: {e}")
        return []

    out: list[dict] = []
    for ev in data.get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        competitors = comp.get("competitors", []) or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_abbr = (home.get("team") or {}).get("abbreviation")
        away_abbr = (away.get("team") or {}).get("abbreviation")
        try:
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
        except (TypeError, ValueError):
            continue
        if not home_abbr or not away_abbr:
            continue
        if home_score > away_score:
            winner = home_abbr
        elif away_score > home_score:
            winner = away_abbr
        else:
            winner = "DRAW"
        out.append({
            "away": away_abbr, "home": home_abbr,
            "awayScore": away_score, "homeScore": home_score,
            "winner": winner,
        })
    return out


def collect_window_results(iso_date: str) -> dict[tuple[str, str], dict]:
    """Pull ESPN results for the prediction date + LOOKAHEAD_DAYS after,
    keyed by (away_abbr, home_abbr) so the lookup is O(1)."""
    base = datetime.strptime(iso_date, "%Y-%m-%d")
    matchups: dict[tuple[str, str], dict] = {}
    for offset in range(LOOKAHEAD_DAYS + 1):
        d = (base + timedelta(days=offset)).strftime("%Y-%m-%d")
        for g in fetch_espn_scoreboard(d):
            key = (g["away"], g["home"])
            # First hit wins (closer to prediction date)
            matchups.setdefault(key, g)
    return matchups


def grade_date(iso_date: str, history: dict) -> int:
    pred_file = PREDICTIONS_DIR / f"{iso_date}.json"
    if not pred_file.exists():
        return 0
    try:
        preds = json.loads(pred_file.read_text())
    except Exception as e:
        print(f"[recap] could not parse {pred_file}: {e}")
        return 0

    picks = preds.get("picks", []) or []
    if not picks:
        return 0

    already = {g["gameId"] for g in history["graded"] if g.get("date") == iso_date}
    espn = collect_window_results(iso_date)
    if not espn:
        return 0

    newly = 0
    for pick in picks:
        gid = pick.get("gameId")
        if gid in already:
            continue
        away_n = normalize_abbr(str(pick.get("away", "")))
        home_n = normalize_abbr(str(pick.get("home", "")))
        result = espn.get((away_n, home_n))
        if not result:
            print(f"[recap] {iso_date} {away_n}@{home_n} not found in ESPN window")
            continue
        picked = normalize_abbr(str(pick.get("pickedTeam", "")))
        actual_winner = result["winner"]
        correct = (picked == actual_winner)  # DRAW marks pick wrong
        history["graded"].append({
            "date": iso_date,
            "gameId": gid,
            "home": pick.get("home"),
            "away": pick.get("away"),
            "pickedTeam": pick.get("pickedTeam"),
            "modelProb": pick.get("modelProb"),
            "actualWinner": actual_winner,
            "correct": correct,
            "homeScore": result["homeScore"],
            "awayScore": result["awayScore"],
            "gradedAt": datetime.now(timezone.utc).isoformat(),
        })
        newly += 1
    return newly


def grade_all_ungraded(history: dict) -> int:
    if not PREDICTIONS_DIR.exists():
        return 0
    total = 0
    today_iso = datetime.now().strftime("%Y-%m-%d")
    for f in sorted(PREDICTIONS_DIR.glob("*.json")):
        iso = f.stem
        if iso >= today_iso:
            continue
        total += grade_date(iso, history)
    return total


def compute_season_stats(history: dict) -> dict:
    graded = history.get("graded", [])
    total = len(graded)
    correct = sum(1 for g in graded if g.get("correct"))
    accuracy = (correct / total) if total > 0 else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy}


# Confidence buckets shown in the Discord embed so the user can see how
# the model's declared probability tracks reality at each tier. Each
# bucket is half-open [lo, hi). modelProb in EPL's grading_history.json
# is the picked-side (3-way max) probability so the lowest possible
# value is ~0.34 in theory — but practically the lowest reportable
# bucket starts at 0.50 since the embed only renders 50%+ tiers.
CONFIDENCE_BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90%+"),
]


def compute_confidence_buckets(history: dict) -> list[dict]:
    """For each confidence bucket return {label, total, correct, accuracy}.

    Only buckets with at least one graded pick are returned so the embed
    doesn't carry empty rows early in the season. modelProb is already
    picked-side (max of home/draw/away) — do NOT take max again."""
    graded = history.get("graded", [])
    out = []
    for lo, hi, label in CONFIDENCE_BUCKETS:
        rows = [g for g in graded
                if g.get("modelProb") is not None
                and lo <= float(g["modelProb"]) < hi]
        if not rows:
            continue
        correct = sum(1 for r in rows if r.get("correct"))
        out.append({
            "label":    label,
            "total":    len(rows),
            "correct":  correct,
            "accuracy": correct / len(rows),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD; grade only this date")
    args = parser.parse_args()

    history = load_history()
    if args.date:
        iso = args.date if "-" in args.date else (
            f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:8]}")
        newly = grade_date(iso, history)
    else:
        newly = grade_all_ungraded(history)

    if newly > 0:
        save_history(history)

    stats = compute_season_stats(history)
    print(f"[recap] newly graded: {newly} picks")
    print(f"[recap] season: {stats['correct']}/{stats['total']} correct "
          f"({stats['accuracy']*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
