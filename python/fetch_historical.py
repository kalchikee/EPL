#!/usr/bin/env python3
"""
EPL Oracle v4.1 — Historical Dataset Builder
Downloads 5 seasons of EPL results from football-data.co.uk (free, no API key),
reconstructs pre-match feature vectors using rolling team stats and Elo ratings
as they would have looked at kick-off, then writes data/training_data.csv.

Requirements:
  pip install pandas numpy requests

Usage: python python/fetch_historical.py
       python python/fetch_historical.py --seasons 3   # only last 3 seasons

football-data.co.uk CSV columns used:
  Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR (H/D/A)
  HST, AST (shots on target, when available)
  B365H, B365D, B365A (Bet365 decimal odds, when available)
"""

import argparse
import csv
import io
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Install with: pip install pandas numpy requests")
    sys.exit(1)

# ─── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent
OUTPUT_PATH = ROOT_DIR / "data" / "training_data.csv"

# ─── Seasons to fetch ─────────────────────────────────────────────────────────
# football-data.co.uk code format: 2020-21 → "2021", url segment "2021/E0.csv"

SEASONS = [
    {"label": "2020-21", "year": 2020, "url_code": "2021"},
    {"label": "2021-22", "year": 2021, "url_code": "2122"},
    {"label": "2022-23", "year": 2022, "url_code": "2223"},
    {"label": "2023-24", "year": 2023, "url_code": "2324"},
    {"label": "2024-25", "year": 2024, "url_code": "2425"},
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# ─── Team name → abbreviation ─────────────────────────────────────────────────

TEAM_MAP = {
    "Arsenal":          "ARS",
    "Aston Villa":      "AVL",
    "Brentford":        "BRE",
    "Brighton":         "BHA",
    "Burnley":          "BUR",
    "Chelsea":          "CHE",
    "Crystal Palace":   "CRY",
    "Everton":          "EVE",
    "Fulham":           "FUL",
    "Leeds":            "LEE",
    "Leeds United":     "LEE",
    "Leicester":        "LEI",
    "Liverpool":        "LIV",
    "Man City":         "MCI",
    "Man United":       "MUN",
    "Newcastle":        "NEW",
    "Nott'm Forest":    "NFO",
    "Nottm Forest":     "NFO",
    "Nottingham Forest":"NFO",
    "Southampton":      "SOU",
    "Tottenham":        "TOT",
    "Watford":          "WAT",
    "West Ham":         "WHU",
    "West Brom":        "WBA",
    "Wolves":           "WOL",
    "Bournemouth":      "BOU",
    "Ipswich":          "IPS",
    "Sheffield United": "SHU",
    "Sheffield Utd":    "SHU",
    "Norwich":          "NOR",
    "Sunderland":       "SUN",
    "Luton":            "LUT",
}

# ─── Constants ─────────────────────────────────────────────────────────────────

LEAGUE_AVG_GOALS = 1.35
HOME_ADV_FACTOR  = 1.18   # must match monteCarlo.ts
ELO_K            = 20
ELO_HOME_ADV     = 75
ELO_START        = 1500
ELO_REGRESSION   = 0.70   # how much prior rating carries into next season
ROLLING_WINDOW   = 10     # matches used for rolling attack/defense/form stats
PRIOR_FADE       = 10     # same as eplClient.ts

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

# ─── Download helpers ──────────────────────────────────────────────────────────

def download_season(url_code: str, label: str) -> pd.DataFrame | None:
    url = BASE_URL.format(code=url_code)
    print(f"  Fetching {label} from {url} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip", encoding="latin-1")
        # Keep only rows with a valid result
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG"])
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        print(f"✓  {len(df)} matches")
        return df
    except Exception as e:
        print(f"✗  {e}")
        return None

def parse_date(d: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(d.strip(), fmt)
        except ValueError:
            continue
    return None

# ─── Elo engine ────────────────────────────────────────────────────────────────

class EloEngine:
    def __init__(self):
        self.ratings: dict[str, float] = defaultdict(lambda: ELO_START)

    def get(self, team: str) -> float:
        return self.ratings[team]

    def diff(self, home: str, away: str) -> float:
        return self.ratings[home] - self.ratings[away]

    def expected(self, home: str, away: str) -> float:
        h = self.ratings[home] + ELO_HOME_ADV
        a = self.ratings[away]
        return 1 / (1 + 10 ** ((a - h) / 400))

    def update(self, home: str, away: str, hg: int, ag: int):
        exp = self.expected(home, away)
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = abs(hg - ag)
        mov = math.log(1 + min(margin, 4)) / math.log(5) if margin > 0 else 0.5
        delta = ELO_K * mov * (actual - exp)
        self.ratings[home] += delta
        self.ratings[away] -= delta

    def regress_season(self):
        for team in list(self.ratings.keys()):
            self.ratings[team] = ELO_REGRESSION * self.ratings[team] + (1 - ELO_REGRESSION) * ELO_START

# ─── Rolling team stats ────────────────────────────────────────────────────────
# Tracks the last ROLLING_WINDOW matches for each team regardless of H/A.

class RollingStats:
    def __init__(self):
        self.gf:  dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.ga:  dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))   # last-5 form
        self.sot: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.cs:  dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.last_match: dict[str, datetime] = {}
        # Home/away form separately
        self.h_pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self.a_pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        # Table: cumulative per season
        self.wins:   dict[str, int] = defaultdict(int)
        self.draws:  dict[str, int] = defaultdict(int)
        self.losses: dict[str, int] = defaultdict(int)
        self.played: dict[str, int] = defaultdict(int)
        self.gf_total: dict[str, int] = defaultdict(int)
        self.ga_total: dict[str, int] = defaultdict(int)

    def reset_season(self):
        self.wins.clear(); self.draws.clear(); self.losses.clear()
        self.played.clear(); self.gf_total.clear(); self.ga_total.clear()

    def _avg(self, dq: deque, default: float) -> float:
        return sum(dq) / len(dq) if dq else default

    def attack_rating(self, team: str, prior: dict) -> float:
        n = len(self.gf[team])
        curr_gf = self._avg(self.gf[team], 0)
        prior_gf = prior.get("gfPg", LEAGUE_AVG_GOALS)
        blended = (n * curr_gf + max(0, PRIOR_FADE - n) * prior_gf) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    def defense_rating(self, team: str, prior: dict) -> float:
        n = len(self.ga[team])
        curr_ga = self._avg(self.ga[team], 0)
        prior_ga = prior.get("gaPg", LEAGUE_AVG_GOALS)
        blended = (n * curr_ga + max(0, PRIOR_FADE - n) * prior_ga) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    def form(self, team: str, prior: dict) -> float:
        n = len(self.pts[team])
        curr = self._avg(self.pts[team], 0) / 3.0
        prior_form = min(1.0, (prior.get("gfPg", LEAGUE_AVG_GOALS) -
                                prior.get("gaPg", LEAGUE_AVG_GOALS) + 1.35) / (1.35 * 2))
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_form) / PRIOR_FADE
        return max(0.0, min(1.0, blended))

    def home_form(self, team: str) -> float:
        return self._avg(self.h_pts[team], 0) / 3.0

    def away_form(self, team: str) -> float:
        return self._avg(self.a_pts[team], 0) / 3.0

    def sot_pg(self, team: str) -> float:
        return self._avg(self.sot[team], 3.5)

    def cs_rate(self, team: str, prior: dict) -> float:
        n = len(self.cs[team])
        curr = self._avg(self.cs[team], 0)
        prior_cs = prior.get("csRate", 0.28)
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_cs) / PRIOR_FADE
        return max(0.0, min(1.0, blended))

    def position(self, team: str, all_teams: list) -> int:
        # Sort all teams by points (desc), GD (desc), GF (desc)
        def sort_key(t):
            p = self.wins[t] * 3 + self.draws[t]
            gd = self.gf_total[t] - self.ga_total[t]
            return (-p, -gd, -self.gf_total[t])
        ranked = sorted(all_teams, key=sort_key)
        try:
            return ranked.index(team) + 1
        except ValueError:
            return 10

    def rest_days(self, team: str, match_date: datetime) -> int:
        last = self.last_match.get(team)
        if not last:
            return 7
        return max(0, (match_date - last).days)

    def update(self, home: str, away: str, hg: int, ag: int,
               h_sot: float, a_sot: float, match_date: datetime):
        result = "H" if hg > ag else "D" if hg == ag else "A"
        h_pts = 3 if result == "H" else 1 if result == "D" else 0
        a_pts = 3 if result == "A" else 1 if result == "D" else 0

        self.gf[home].append(hg); self.ga[home].append(ag)
        self.gf[away].append(ag); self.ga[away].append(hg)
        self.pts[home].append(h_pts); self.pts[away].append(a_pts)
        self.h_pts[home].append(h_pts); self.a_pts[away].append(a_pts)
        self.sot[home].append(h_sot); self.sot[away].append(a_sot)
        self.cs[home].append(1 if ag == 0 else 0)
        self.cs[away].append(1 if hg == 0 else 0)

        self.played[home] += 1; self.played[away] += 1
        self.gf_total[home] += hg; self.ga_total[home] += ag
        self.gf_total[away] += ag; self.ga_total[away] += hg
        if result == "H":  self.wins[home]  += 1; self.losses[away] += 1
        elif result == "A":self.wins[away]  += 1; self.losses[home] += 1
        else:              self.draws[home] += 1; self.draws[away]  += 1

        self.last_match[home] = match_date
        self.last_match[away] = match_date

# ─── Odds helpers ──────────────────────────────────────────────────────────────

def decimal_to_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1 else 0.0

def remove_vig(h: float, d: float, a: float):
    total = h + d + a
    if total <= 0:
        return (0.45, 0.27, 0.28)
    return (h / total, d / total, a / total)

# ─── Poisson PMF ──────────────────────────────────────────────────────────────

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = k * math.log(lam) - lam - sum(math.log(i) for i in range(1, k + 1))
    return math.exp(log_p)

RHO = -0.15  # Dixon-Coles

def dc_tau(h: int, a: int, lh: float, la: float) -> float:
    if h == 0 and a == 0: return 1 - lh * la * RHO
    if h == 1 and a == 0: return 1 + la * RHO
    if h == 0 and a == 1: return 1 + lh * RHO
    if h == 1 and a == 1: return 1 - RHO
    return 1.0

def poisson_probs(lh: float, la: float, max_g: int = 8):
    hw = dw = aw = 0.0
    total = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson_pmf(h, lh) * poisson_pmf(a, la) * dc_tau(h, a, lh, la)
            p = max(0.0, p)
            total += p
            if h > a:  hw += p
            elif h == a: dw += p
            else:       aw += p
    if total > 0:
        hw /= total; dw /= total; aw /= total
    return max(0.01, hw), max(0.01, dw), max(0.01, aw)

# ─── Prior stats (must match eplClient.ts PRIOR_STATS_2024_25) ────────────────

PRIOR_STATS: dict[str, dict] = {
    "LIV": {"gfPg": 2.34, "gaPg": 0.97, "csRate": 0.45},
    "ARS": {"gfPg": 1.84, "gaPg": 0.97, "csRate": 0.42},
    "MCI": {"gfPg": 1.82, "gaPg": 1.37, "csRate": 0.34},
    "CHE": {"gfPg": 1.68, "gaPg": 1.42, "csRate": 0.31},
    "NEW": {"gfPg": 1.68, "gaPg": 1.21, "csRate": 0.37},
    "AVL": {"gfPg": 1.61, "gaPg": 1.47, "csRate": 0.29},
    "TOT": {"gfPg": 1.74, "gaPg": 1.58, "csRate": 0.26},
    "NFO": {"gfPg": 1.16, "gaPg": 1.21, "csRate": 0.37},
    "BHA": {"gfPg": 1.58, "gaPg": 1.50, "csRate": 0.29},
    "FUL": {"gfPg": 1.50, "gaPg": 1.53, "csRate": 0.27},
    "MUN": {"gfPg": 1.18, "gaPg": 1.61, "csRate": 0.24},
    "WOL": {"gfPg": 1.37, "gaPg": 1.74, "csRate": 0.21},
    "BRE": {"gfPg": 1.37, "gaPg": 1.76, "csRate": 0.21},
    "CRY": {"gfPg": 1.11, "gaPg": 1.55, "csRate": 0.26},
    "EVE": {"gfPg": 1.13, "gaPg": 1.47, "csRate": 0.28},
    "WHU": {"gfPg": 1.34, "gaPg": 1.74, "csRate": 0.21},
    "BOU": {"gfPg": 1.53, "gaPg": 1.63, "csRate": 0.26},
    "LEE": {"gfPg": 1.27, "gaPg": 1.57, "csRate": 0.22},
    "BUR": {"gfPg": 1.16, "gaPg": 1.62, "csRate": 0.21},
    "SUN": {"gfPg": 1.13, "gaPg": 1.60, "csRate": 0.22},
    # Teams from earlier seasons
    "LEI": {"gfPg": 1.18, "gaPg": 1.79, "csRate": 0.18},
    "SOU": {"gfPg": 0.76, "gaPg": 2.21, "csRate": 0.13},
    "IPS": {"gfPg": 0.97, "gaPg": 1.84, "csRate": 0.18},
    "WAT": {"gfPg": 1.11, "gaPg": 1.68, "csRate": 0.21},
    "NOR": {"gfPg": 0.92, "gaPg": 1.84, "csRate": 0.18},
    "WBA": {"gfPg": 0.87, "gaPg": 1.66, "csRate": 0.21},
    "SHU": {"gfPg": 0.76, "gaPg": 2.08, "csRate": 0.13},
    "LUT": {"gfPg": 0.89, "gaPg": 1.76, "csRate": 0.18},
}

DEFAULT_PRIOR = {"gfPg": LEAGUE_AVG_GOALS, "gaPg": LEAGUE_AVG_GOALS, "csRate": 0.28}

# ─── Main ─────────────────────────────────────────────────────────────────────

def build_gameweek_map(df: pd.DataFrame) -> dict:
    """Assign approximate gameweek numbers by sorting matches by date."""
    dates = sorted(df["parsed_date"].dropna().unique())
    # Group dates into ~10 GW clusters (10 games per GW across ~7 days)
    gw_map = {}
    gw = 1
    prev_date = None
    for d in dates:
        if prev_date and (d - prev_date).days > 10:
            gw += 1
        gw_map[d] = gw
        prev_date = d
    return gw_map

def run(num_seasons: int = 5):
    seasons_to_use = SEASONS[-num_seasons:]
    print(f"\n[INFO] Fetching {len(seasons_to_use)} EPL seasons from football-data.co.uk\n")

    all_rows = []
    elo    = EloEngine()
    stats  = RollingStats()

    for season_meta in seasons_to_use:
        label    = season_meta["label"]
        year     = season_meta["year"]
        url_code = season_meta["url_code"]

        print(f"── Season {label} ──")

        df = download_season(url_code, label)
        if df is None:
            print(f"  Skipping {label}")
            continue

        # Parse dates
        df["parsed_date"] = df["Date"].apply(parse_date)
        df = df.dropna(subset=["parsed_date"])
        df = df.sort_values("parsed_date").reset_index(drop=True)

        # Map team names to abbreviations
        df["home_abbr"] = df["HomeTeam"].map(TEAM_MAP)
        df["away_abbr"] = df["AwayTeam"].map(TEAM_MAP)
        unmapped = df[df["home_abbr"].isna() | df["away_abbr"].isna()]["HomeTeam"].unique()
        if len(unmapped):
            print(f"  [WARN] Unmapped teams: {list(unmapped)[:5]}")
        df = df.dropna(subset=["home_abbr", "away_abbr"])

        # Regress Elo at season start
        elo.regress_season()
        stats.reset_season()

        # Build gameweek approximation
        gw_map = build_gameweek_map(df)

        all_teams = list(df["home_abbr"].unique())

        skipped = 0
        for _, row in df.iterrows():
            home = row["home_abbr"]
            away = row["away_abbr"]
            hg   = int(row["FTHG"])
            ag   = int(row["FTAG"])
            ftr  = str(row["FTR"]).strip()
            date = row["parsed_date"]
            gw   = gw_map.get(date, 0)

            # ── Pre-match features (computed BEFORE updating stats) ────────────

            prior_h = PRIOR_STATS.get(home, DEFAULT_PRIOR)
            prior_a = PRIOR_STATS.get(away, DEFAULT_PRIOR)

            att_h = stats.attack_rating(home, prior_h)
            att_a = stats.attack_rating(away, prior_a)
            def_h = stats.defense_rating(home, prior_h)
            def_a = stats.defense_rating(away, prior_a)

            gf_h  = att_h * LEAGUE_AVG_GOALS
            gf_a  = att_a * LEAGUE_AVG_GOALS
            ga_h  = def_h * LEAGUE_AVG_GOALS
            ga_a  = def_a * LEAGUE_AVG_GOALS

            form_h = stats.form(home, prior_h)
            form_a = stats.form(away, prior_a)
            hform  = stats.home_form(home)
            aform  = stats.away_form(away)

            pos_h = stats.position(home, all_teams)
            pos_a = stats.position(away, all_teams)

            sot_h = stats.sot_pg(home)
            sot_a = stats.sot_pg(away)

            cs_h = stats.cs_rate(home, prior_h)
            cs_a = stats.cs_rate(away, prior_a)

            rest_h = stats.rest_days(home, date)
            rest_a = stats.rest_days(away, date)

            lam_h = LEAGUE_AVG_GOALS * att_h * def_a * HOME_ADV_FACTOR
            lam_a = LEAGUE_AVG_GOALS * att_a * def_h
            lam_h = max(0.20, min(4.5, lam_h))
            lam_a = max(0.20, min(4.5, lam_a))

            mc_h, mc_d, mc_a = poisson_probs(lam_h, lam_a)

            # Vegas odds (Bet365 if available)
            veg_h = veg_d = veg_a = 0.0
            try:
                b365h = float(row.get("B365H", 0) or 0)
                b365d = float(row.get("B365D", 0) or 0)
                b365a = float(row.get("B365A", 0) or 0)
                if b365h > 1 and b365d > 1 and b365a > 1:
                    raw_h = decimal_to_prob(b365h)
                    raw_d = decimal_to_prob(b365d)
                    raw_a = decimal_to_prob(b365a)
                    veg_h, veg_d, veg_a = remove_vig(raw_h, raw_d, raw_a)
            except (TypeError, ValueError):
                pass

            # Shots on target (if column exists)
            try:
                h_sot_val = float(row.get("HST", 0) or 0)
                a_sot_val = float(row.get("AST", 0) or 0)
            except (TypeError, ValueError):
                h_sot_val = a_sot_val = 3.5

            # ── Assemble feature vector ────────────────────────────────────────

            fv = {
                "elo_diff":              elo.diff(home, away),
                "attack_diff":           att_h - att_a,
                "defense_diff":          def_a - def_h,   # positive = home better def
                "goal_diff_diff":        (gf_h - ga_h) - (gf_a - ga_a),
                "goals_for_diff":        gf_h - gf_a,
                "goals_against_diff":    ga_h - ga_a,
                "net_goals_diff":        (gf_h - ga_h) - (gf_a - ga_a),
                "form_diff":             form_h - form_a,
                "home_form":             hform,
                "away_form":             aform,
                "position_diff":         pos_h - pos_a,
                "shots_on_target_diff":  sot_h - sot_a,
                "possession_diff":       0.0,   # not in simple CSV
                "clean_sheet_diff":      cs_h - cs_a,
                "rest_days_diff":        rest_h - rest_a,
                "home_short_rest":       1 if rest_h <= 3 else 0,
                "away_short_rest":       1 if rest_a <= 3 else 0,
                "home_euro_fatigue":     1 if rest_h <= 3 else 0,
                "away_euro_fatigue":     1 if rest_a <= 3 else 0,
                "is_neutral":            0,
                "lambda_home":           lam_h,
                "lambda_away":           lam_a,
                "vegas_home_prob":       veg_h,
                "vegas_draw_prob":       veg_d,
                "mc_home_win_prob":      mc_h,
            }

            outcome = "home" if ftr == "H" else "draw" if ftr == "D" else "away"
            match_id = f"{year}-{date.strftime('%Y%m%d')}-{home}-{away}"

            all_rows.append({
                **{k: fv[k] for k in FEATURE_NAMES},
                "match_id":       match_id,
                "gameweek":       gw,
                "season":         year,
                "home_team":      home,
                "away_team":      away,
                "match_date":     date.strftime("%Y-%m-%d"),
                "actual_outcome": outcome,
                "home_win_label": 1 if outcome == "home" else 0,
                "draw_label":     1 if outcome == "draw" else 0,
                "away_win_label": 1 if outcome == "away" else 0,
            })

            # ── Update stats (AFTER building features) ─────────────────────────
            elo.update(home, away, hg, ag)
            stats.update(home, away, hg, ag, h_sot_val, a_sot_val, date)

        season_rows = [r for r in all_rows if r["season"] == year]
        h = sum(1 for r in season_rows if r["actual_outcome"] == "home")
        d = sum(1 for r in season_rows if r["actual_outcome"] == "draw")
        a = sum(1 for r in season_rows if r["actual_outcome"] == "away")
        n = len(season_rows)
        print(f"  Processed {n} matches  H:{h/n*100:.0f}%  D:{d/n*100:.0f}%  A:{a/n*100:.0f}%")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = FEATURE_NAMES + ["match_id", "gameweek", "season", "home_team", "away_team",
                               "match_date", "actual_outcome", "home_win_label", "draw_label", "away_win_label"]

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    n = len(all_rows)
    h = sum(1 for r in all_rows if r["actual_outcome"] == "home")
    d = sum(1 for r in all_rows if r["actual_outcome"] == "draw")
    a = sum(1 for r in all_rows if r["actual_outcome"] == "away")

    print(f"\n[OK] {n} total matches written to {OUTPUT_PATH}")
    print(f"     H:{h} ({h/n*100:.0f}%)  D:{d} ({d/n*100:.0f}%)  A:{a} ({a/n*100:.0f}%)")
    print(f"\nNext step: python python/train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=5,
                        help="Number of most recent seasons to fetch (default: 5)")
    args = parser.parse_args()
    run(num_seasons=min(args.seasons, len(SEASONS)))
