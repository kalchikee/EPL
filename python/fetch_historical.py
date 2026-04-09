#!/usr/bin/env python3
"""
EPL Oracle v4.2 — Historical Dataset Builder
Downloads 10 seasons of EPL results from football-data.co.uk (free, no API key),
reconstructs pre-match feature vectors using rolling team stats and Elo ratings
as they would have looked at kick-off, then writes data/training_data.csv.

Improvements over v4.1:
  - 10 seasons instead of 5 (doubles training data)
  - Exponential decay in rolling windows (recent games count more)
  - Home/away split attack/defense ratings (4 new features)
  - Multiple bookmaker average (B365, BW, VC, PS) for better odds signal

Requirements:
  pip install pandas numpy requests

Usage: python python/fetch_historical.py
       python python/fetch_historical.py --seasons 10
"""

import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

try:
    import requests
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Install with: pip install pandas numpy requests")
    sys.exit(1)

# Paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent
OUTPUT_PATH = ROOT_DIR / "data" / "training_data.csv"

# 10 seasons of EPL data (football-data.co.uk code format)
SEASONS = [
    {"label": "2015-16", "year": 2015, "url_code": "1516"},
    {"label": "2016-17", "year": 2016, "url_code": "1617"},
    {"label": "2017-18", "year": 2017, "url_code": "1718"},
    {"label": "2018-19", "year": 2018, "url_code": "1819"},
    {"label": "2019-20", "year": 2019, "url_code": "1920"},
    {"label": "2020-21", "year": 2020, "url_code": "2021"},
    {"label": "2021-22", "year": 2021, "url_code": "2122"},
    {"label": "2022-23", "year": 2022, "url_code": "2223"},
    {"label": "2023-24", "year": 2023, "url_code": "2324"},
    {"label": "2024-25", "year": 2024, "url_code": "2425"},
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

TEAM_MAP = {
    "Arsenal":           "ARS",
    "Aston Villa":       "AVL",
    "Brentford":         "BRE",
    "Brighton":          "BHA",
    "Burnley":           "BUR",
    "Chelsea":           "CHE",
    "Crystal Palace":    "CRY",
    "Everton":           "EVE",
    "Fulham":            "FUL",
    "Hull":              "HUL",
    "Huddersfield":      "HUD",
    "Leeds":             "LEE",
    "Leeds United":      "LEE",
    "Leicester":         "LEI",
    "Liverpool":         "LIV",
    "Man City":          "MCI",
    "Man United":        "MUN",
    "Middlesbrough":     "MID",
    "Newcastle":         "NEW",
    "Nott'm Forest":     "NFO",
    "Nottm Forest":      "NFO",
    "Nottingham Forest": "NFO",
    "Southampton":       "SOU",
    "Stoke":             "STK",
    "Swansea":           "SWA",
    "Sunderland":        "SUN",
    "Tottenham":         "TOT",
    "Watford":           "WAT",
    "West Ham":          "WHU",
    "West Brom":         "WBA",
    "Wolves":            "WOL",
    "Bournemouth":       "BOU",
    "Ipswich":           "IPS",
    "Sheffield United":  "SHU",
    "Sheffield Utd":     "SHU",
    "Norwich":           "NOR",
    "Luton":             "LUT",
    "Cardiff":           "CAR",
}

# Constants — must match monteCarlo.ts and eplClient.ts
LEAGUE_AVG_GOALS = 1.35
ELO_K            = 20
ELO_HOME_ADV     = 75
ELO_START        = 1500
ELO_REGRESSION   = 0.70
ROLLING_WINDOW   = 10
PRIOR_FADE       = 10
EXP_DECAY_ALPHA  = 0.30   # exponential decay: recent games count more

# Home/away adjustment factors for prior estimation
HOME_ATT_FACTOR  = 1.18   # home teams score ~18% more at home
HOME_DEF_FACTOR  = 0.88   # home teams concede ~12% less at home

FEATURE_NAMES = [
    "elo_diff", "attack_diff", "defense_diff", "goal_diff_diff",
    "goals_for_diff", "goals_against_diff", "net_goals_diff",
    "form_diff", "home_form", "away_form",
    "position_diff", "shots_on_target_diff", "possession_diff", "clean_sheet_diff",
    "rest_days_diff", "home_short_rest", "away_short_rest",
    "home_euro_fatigue", "away_euro_fatigue", "is_neutral",
    "lambda_home", "lambda_away",
    "vegas_home_prob", "vegas_draw_prob", "mc_home_win_prob",
    # v4.2: home/away split attack/defense
    "home_att_home", "home_def_home", "away_att_away", "away_def_away",
    # v4.2: head-to-head history
    "h2h_home_win_rate", "h2h_goal_diff",
    # v4.3: closing line signal + corners + referee bias
    "line_movement_home", "corners_diff", "referee_home_bias",
]

# EPL historical home win rate (default for teams with no H2H history)
H2H_DEFAULT_WIN_RATE = 0.44
H2H_DEFAULT_GOAL_DIFF = 0.28

# ── Download helpers ──────────────────────────────────────────────────────────

def download_season(url_code: str, label: str) -> pd.DataFrame | None:
    url = BASE_URL.format(code=url_code)
    print(f"  Fetching {label} from {url} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip", encoding="latin-1")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG"])
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        print(f"OK  {len(df)} matches")
        return df
    except Exception as e:
        print(f"FAIL  {e}")
        return None

def parse_date(d: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(d.strip(), fmt)
        except ValueError:
            continue
    return None

# ── Elo engine ────────────────────────────────────────────────────────────────

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

# ── Rolling stats with exponential decay ──────────────────────────────────────

class RollingStats:
    def __init__(self):
        # Overall rolling windows (home + away combined)
        self.gf:  dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.ga:  dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self.sot: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.cs:  dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        # Home-specific windows
        self.h_gf: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.h_ga: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        # Away-specific windows
        self.a_gf: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.a_ga: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        # Home/away form
        self.h_pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        self.a_pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
        # Corners per game
        self.corners: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        # Table
        self.wins:    dict[str, int] = defaultdict(int)
        self.draws:   dict[str, int] = defaultdict(int)
        self.losses:  dict[str, int] = defaultdict(int)
        self.played:  dict[str, int] = defaultdict(int)
        self.gf_tot:  dict[str, int] = defaultdict(int)
        self.ga_tot:  dict[str, int] = defaultdict(int)
        self.last_match: dict[str, datetime] = {}

    def reset_season(self):
        for d in [self.wins, self.draws, self.losses, self.played, self.gf_tot, self.ga_tot]:
            d.clear()

    def _exp_avg(self, dq: deque, default: float = 0.0) -> float:
        """Exponentially weighted average — recent games count more."""
        if not dq:
            return default
        items = list(dq)
        n = len(items)
        alpha = EXP_DECAY_ALPHA
        weights = [alpha * (1 - alpha) ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        return sum(w * v for w, v in zip(weights, items)) / total_w

    # ── Overall ratings ────────────────────────────────────────────────────────

    def attack_rating(self, team: str, prior: dict) -> float:
        n = len(self.gf[team])
        curr = self._exp_avg(self.gf[team], 0)
        prior_gf = prior.get("gfPg", LEAGUE_AVG_GOALS)
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_gf) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    def defense_rating(self, team: str, prior: dict) -> float:
        n = len(self.ga[team])
        curr = self._exp_avg(self.ga[team], 0)
        prior_ga = prior.get("gaPg", LEAGUE_AVG_GOALS)
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_ga) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    # ── Home/away split ratings (new in v4.2) ─────────────────────────────────

    def home_attack_rating(self, team: str, prior: dict) -> float:
        """Attack rating for home games only."""
        n = len(self.h_gf[team])
        curr = self._exp_avg(self.h_gf[team], 0)
        prior_home_gf = prior.get("gfPg", LEAGUE_AVG_GOALS) * HOME_ATT_FACTOR
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_home_gf) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    def home_defense_rating(self, team: str, prior: dict) -> float:
        """Defense rating (GA) for home games only — lower = better."""
        n = len(self.h_ga[team])
        curr = self._exp_avg(self.h_ga[team], 0)
        prior_home_ga = prior.get("gaPg", LEAGUE_AVG_GOALS) * HOME_DEF_FACTOR  # less GA at home
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_home_ga) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    def away_attack_rating(self, team: str, prior: dict) -> float:
        """Attack rating for away games only."""
        n = len(self.a_gf[team])
        curr = self._exp_avg(self.a_gf[team], 0)
        prior_away_gf = prior.get("gfPg", LEAGUE_AVG_GOALS) / HOME_ATT_FACTOR
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_away_gf) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    def away_defense_rating(self, team: str, prior: dict) -> float:
        """Defense rating (GA) for away games only — higher = worse defense."""
        n = len(self.a_ga[team])
        curr = self._exp_avg(self.a_ga[team], 0)
        prior_away_ga = prior.get("gaPg", LEAGUE_AVG_GOALS) / HOME_DEF_FACTOR  # more GA away
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_away_ga) / PRIOR_FADE
        return max(0.3, blended / LEAGUE_AVG_GOALS)

    # ── Form / misc ───────────────────────────────────────────────────────────

    def form(self, team: str, prior: dict) -> float:
        n = len(self.pts[team])
        curr = self._exp_avg(self.pts[team], 0) / 3.0
        prior_form = min(1.0, (prior.get("gfPg", LEAGUE_AVG_GOALS) -
                                prior.get("gaPg", LEAGUE_AVG_GOALS) + 1.35) / (1.35 * 2))
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_form) / PRIOR_FADE
        return max(0.0, min(1.0, blended))

    def home_form(self, team: str) -> float:
        return self._exp_avg(self.h_pts[team], 0) / 3.0

    def away_form(self, team: str) -> float:
        return self._exp_avg(self.a_pts[team], 0) / 3.0

    def sot_pg(self, team: str) -> float:
        return self._exp_avg(self.sot[team], 3.5)

    def corners_pg(self, team: str) -> float:
        return self._exp_avg(self.corners[team], 5.0)

    def cs_rate(self, team: str, prior: dict) -> float:
        n = len(self.cs[team])
        curr = self._exp_avg(self.cs[team], 0)
        prior_cs = prior.get("csRate", 0.28)
        blended = (n * curr + max(0, PRIOR_FADE - n) * prior_cs) / PRIOR_FADE
        return max(0.0, min(1.0, blended))

    def position(self, team: str, all_teams: list) -> int:
        def sort_key(t):
            p  = self.wins[t] * 3 + self.draws[t]
            gd = self.gf_tot[t] - self.ga_tot[t]
            return (-p, -gd, -self.gf_tot[t])
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
               h_sot: float, a_sot: float, match_date: datetime,
               h_corners: float = 5.0, a_corners: float = 5.0):
        result = "H" if hg > ag else "D" if hg == ag else "A"
        h_pts = 3 if result == "H" else 1 if result == "D" else 0
        a_pts = 3 if result == "A" else 1 if result == "D" else 0

        # Overall windows
        self.gf[home].append(hg); self.ga[home].append(ag)
        self.gf[away].append(ag); self.ga[away].append(hg)
        self.pts[home].append(h_pts); self.pts[away].append(a_pts)
        self.sot[home].append(h_sot); self.sot[away].append(a_sot)
        self.cs[home].append(1 if ag == 0 else 0)
        self.cs[away].append(1 if hg == 0 else 0)
        # Home/away split windows
        self.h_gf[home].append(hg); self.h_ga[home].append(ag)
        self.a_gf[away].append(ag); self.a_ga[away].append(hg)
        # Form
        self.h_pts[home].append(h_pts); self.a_pts[away].append(a_pts)
        # Table
        self.played[home] += 1; self.played[away] += 1
        self.gf_tot[home] += hg; self.ga_tot[home] += ag
        self.gf_tot[away] += ag; self.ga_tot[away] += hg
        if result == "H":   self.wins[home]   += 1; self.losses[away] += 1
        elif result == "A": self.wins[away]   += 1; self.losses[home] += 1
        else:               self.draws[home]  += 1; self.draws[away]  += 1
        self.corners[home].append(h_corners)
        self.corners[away].append(a_corners)
        self.last_match[home] = match_date
        self.last_match[away] = match_date

# ── Odds helpers ───────────────────────────────────────────────────────────────

def decimal_to_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1 else 0.0

def remove_vig(h: float, d: float, a: float):
    total = h + d + a
    if total <= 0:
        return (0.45, 0.27, 0.28)
    return (h / total, d / total, a / total)

def get_bookmaker_avg(row) -> tuple[float, float, float]:
    """Average multiple opening bookmakers' implied probabilities (after vig removal)."""
    books = [
        ("B365H", "B365D", "B365A"),
        ("BWH",   "BWD",   "BWA"),
        ("VCH",   "VCD",   "VCA"),
        ("PSH",   "PSD",   "PSA"),
        ("WHH",   "WHD",   "WHA"),
    ]
    valid_probs = []
    for hk, dk, ak in books:
        try:
            h = float(row.get(hk) or 0)
            d = float(row.get(dk) or 0)
            a = float(row.get(ak) or 0)
            if h > 1 and d > 1 and a > 1:
                ph = decimal_to_prob(h)
                pd_ = decimal_to_prob(d)
                pa = decimal_to_prob(a)
                vp, vd, va = remove_vig(ph, pd_, pa)
                valid_probs.append((vp, vd, va))
        except (TypeError, ValueError):
            continue

    if not valid_probs:
        return (0.0, 0.0, 0.0)

    avg_h = float(np.mean([p[0] for p in valid_probs]))
    avg_d = float(np.mean([p[1] for p in valid_probs]))
    avg_a = float(np.mean([p[2] for p in valid_probs]))
    return (avg_h, avg_d, avg_a)

def get_closing_avg(row) -> tuple[float, float, float]:
    """Market-average CLOSING odds (AvgCH/AvgCD/AvgCA) — sharper than opening lines."""
    try:
        ch = float(row.get("AvgCH") or 0)
        cd = float(row.get("AvgCD") or 0)
        ca = float(row.get("AvgCA") or 0)
        if ch > 1 and cd > 1 and ca > 1:
            ph = decimal_to_prob(ch)
            pd_ = decimal_to_prob(cd)
            pa = decimal_to_prob(ca)
            return remove_vig(ph, pd_, pa)
    except (TypeError, ValueError):
        pass
    # Fallback: try Avg (opening market average)
    try:
        oh = float(row.get("AvgH") or 0)
        od = float(row.get("AvgD") or 0)
        oa = float(row.get("AvgA") or 0)
        if oh > 1 and od > 1 and oa > 1:
            ph = decimal_to_prob(oh)
            pd_ = decimal_to_prob(od)
            pa = decimal_to_prob(oa)
            return remove_vig(ph, pd_, pa)
    except (TypeError, ValueError):
        pass
    return (0.0, 0.0, 0.0)

# ── Poisson / Dixon-Coles ──────────────────────────────────────────────────────

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = k * math.log(lam) - lam - sum(math.log(i) for i in range(1, k + 1))
    return math.exp(log_p)

RHO = -0.15

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

# ── Prior stats ────────────────────────────────────────────────────────────────

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
    "LEI": {"gfPg": 1.18, "gaPg": 1.79, "csRate": 0.18},
    "SOU": {"gfPg": 0.76, "gaPg": 2.21, "csRate": 0.13},
    "IPS": {"gfPg": 0.97, "gaPg": 1.84, "csRate": 0.18},
    "WAT": {"gfPg": 1.11, "gaPg": 1.68, "csRate": 0.21},
    "NOR": {"gfPg": 0.92, "gaPg": 1.84, "csRate": 0.18},
    "WBA": {"gfPg": 0.87, "gaPg": 1.66, "csRate": 0.21},
    "SHU": {"gfPg": 0.76, "gaPg": 2.08, "csRate": 0.13},
    "LUT": {"gfPg": 0.89, "gaPg": 1.76, "csRate": 0.18},
    # Additional older teams
    "HUL": {"gfPg": 1.05, "gaPg": 1.74, "csRate": 0.21},
    "HUD": {"gfPg": 0.89, "gaPg": 1.71, "csRate": 0.18},
    "STK": {"gfPg": 1.08, "gaPg": 1.66, "csRate": 0.22},
    "SWA": {"gfPg": 1.11, "gaPg": 1.58, "csRate": 0.24},
    "MID": {"gfPg": 1.00, "gaPg": 1.58, "csRate": 0.24},
    "CAR": {"gfPg": 0.89, "gaPg": 1.76, "csRate": 0.18},
}

DEFAULT_PRIOR = {"gfPg": LEAGUE_AVG_GOALS, "gaPg": LEAGUE_AVG_GOALS, "csRate": 0.28}

# ── GW date approximation ─────────────────────────────────────────────────────

def build_gameweek_map(df: pd.DataFrame) -> dict:
    dates = sorted(df["parsed_date"].dropna().unique())
    gw_map = {}
    gw = 1
    prev_date = None
    for d in dates:
        if prev_date and (d - prev_date).days > 10:
            gw += 1
        gw_map[d] = gw
        prev_date = d
    return gw_map

# ── Main ──────────────────────────────────────────────────────────────────────

def run(num_seasons: int = 10):
    seasons_to_use = SEASONS[-num_seasons:]
    print(f"\n[INFO] Fetching {len(seasons_to_use)} EPL seasons from football-data.co.uk\n")

    all_rows = []
    elo   = EloEngine()
    stats = RollingStats()
    # H2H records: key = frozenset({team1, team2}), persists across all seasons
    h2h_records: dict = defaultdict(list)  # {(team_a, team_b): [{home, away, hg, ag, date}]}
    # Referee home bias tracking: ref_name -> {home_wins, total}
    referee_stats: dict[str, dict] = defaultdict(lambda: {"home_wins": 0, "total": 0})

    for season_meta in seasons_to_use:
        label    = season_meta["label"]
        year     = season_meta["year"]
        url_code = season_meta["url_code"]

        print(f"-- Season {label} --")

        df = download_season(url_code, label)
        if df is None:
            print(f"  Skipping {label}")
            continue

        df["parsed_date"] = df["Date"].apply(parse_date)
        df = df.dropna(subset=["parsed_date"])
        df = df.sort_values("parsed_date").reset_index(drop=True)

        df["home_abbr"] = df["HomeTeam"].map(TEAM_MAP)
        df["away_abbr"] = df["AwayTeam"].map(TEAM_MAP)
        unmapped = df[df["home_abbr"].isna() | df["away_abbr"].isna()]["HomeTeam"].unique()
        if len(unmapped):
            print(f"  [WARN] Unmapped teams: {list(unmapped)[:5]}")
        df = df.dropna(subset=["home_abbr", "away_abbr"])

        elo.regress_season()
        stats.reset_season()

        gw_map    = build_gameweek_map(df)
        all_teams = list(df["home_abbr"].unique())

        for _, row in df.iterrows():
            home = row["home_abbr"]
            away = row["away_abbr"]
            hg   = int(row["FTHG"])
            ag   = int(row["FTAG"])
            ftr  = str(row["FTR"]).strip()
            date = row["parsed_date"]
            gw   = gw_map.get(date, 0)

            prior_h = PRIOR_STATS.get(home, DEFAULT_PRIOR)
            prior_a = PRIOR_STATS.get(away, DEFAULT_PRIOR)

            # Overall ratings
            att_h = stats.attack_rating(home, prior_h)
            att_a = stats.attack_rating(away, prior_a)
            def_h = stats.defense_rating(home, prior_h)
            def_a = stats.defense_rating(away, prior_a)

            gf_h = att_h * LEAGUE_AVG_GOALS
            gf_a = att_a * LEAGUE_AVG_GOALS
            ga_h = def_h * LEAGUE_AVG_GOALS
            ga_a = def_a * LEAGUE_AVG_GOALS

            # Home/away split ratings (new features)
            h_att_home = stats.home_attack_rating(home, prior_h)    # home team at home
            h_def_home = stats.home_defense_rating(home, prior_h)   # home team def at home
            a_att_away = stats.away_attack_rating(away, prior_a)    # away team away
            a_def_away = stats.away_defense_rating(away, prior_a)   # away team def away

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

            # Updated lambdas using home/away specific ratings (no HOME_ADV_FACTOR needed)
            # home scores: home team's HOME attack vs away team's AWAY defense
            # away scores: away team's AWAY attack vs home team's HOME defense
            lam_h = LEAGUE_AVG_GOALS * h_att_home * a_def_away
            lam_a = LEAGUE_AVG_GOALS * a_att_away * h_def_home
            lam_h = max(0.20, min(4.5, lam_h))
            lam_a = max(0.20, min(4.5, lam_a))

            mc_h, mc_d, mc_a = poisson_probs(lam_h, lam_a)

            # Opening bookmaker average (B365, BW, VC, PS, WH)
            opening_h, opening_d, opening_a = get_bookmaker_avg(row)

            # Closing market average (AvgCH/CD/CA) — sharper than opening lines
            closing_h, closing_d, closing_a = get_closing_avg(row)

            # Use closing odds as the vegas signal if available, else opening
            if closing_h > 0:
                veg_h, veg_d, veg_a = closing_h, closing_d, closing_a
            else:
                veg_h, veg_d, veg_a = opening_h, opening_d, opening_a

            # Line movement: closing prob - opening prob (positive = sharp money on home)
            if closing_h > 0 and opening_h > 0:
                line_movement_home = float(closing_h - opening_h)
            else:
                line_movement_home = 0.0
            line_movement_home = max(-0.20, min(0.20, line_movement_home))

            # Shots on target
            try:
                h_sot_val = float(row.get("HST", 0) or 0)
                a_sot_val = float(row.get("AST", 0) or 0)
            except (TypeError, ValueError):
                h_sot_val = a_sot_val = 3.5

            # Corners
            try:
                h_corners_val = float(row.get("HC", 0) or 0)
                a_corners_val = float(row.get("AC", 0) or 0)
            except (TypeError, ValueError):
                h_corners_val = a_corners_val = 5.0

            # Corners rolling differential
            corners_diff = stats.corners_pg(home) - stats.corners_pg(away)

            # Referee home bias
            ref_name = str(row.get("Referee", "")).strip()
            if ref_name and ref_name.lower() != "nan" and ref_name != "":
                ref = referee_stats[ref_name]
                if ref["total"] >= 10:
                    ref_bias = (ref["home_wins"] / ref["total"]) - 0.44
                else:
                    ref_bias = 0.0
            else:
                ref_bias = 0.0

            # ── Head-to-head features ──────────────────────────────────────────
            pair_key = tuple(sorted([home, away]))
            recent_h2h = h2h_records[pair_key][-5:]
            if recent_h2h:
                from_home_perspective = [
                    (m["hg"] if m["home"] == home else m["ag"],   # home team goals
                     m["ag"] if m["home"] == home else m["hg"])   # away team goals
                    for m in recent_h2h
                ]
                h2h_wins = sum(1 for h_g, a_g in from_home_perspective if h_g > a_g)
                h2h_home_win_rate = h2h_wins / len(recent_h2h)
                h2h_goal_diff = float(np.mean([h_g - a_g for h_g, a_g in from_home_perspective]))
            else:
                h2h_home_win_rate = H2H_DEFAULT_WIN_RATE
                h2h_goal_diff = H2H_DEFAULT_GOAL_DIFF

            fv = {
                "elo_diff":              elo.diff(home, away),
                "attack_diff":           att_h - att_a,
                "defense_diff":          def_a - def_h,
                "goal_diff_diff":        (gf_h - ga_h) - (gf_a - ga_a),
                "goals_for_diff":        gf_h - gf_a,
                "goals_against_diff":    ga_h - ga_a,
                "net_goals_diff":        (gf_h - ga_h) - (gf_a - ga_a),
                "form_diff":             form_h - form_a,
                "home_form":             hform,
                "away_form":             aform,
                "position_diff":         pos_h - pos_a,
                "shots_on_target_diff":  sot_h - sot_a,
                "possession_diff":       0.0,
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
                # v4.2: home/away split
                "home_att_home":         h_att_home,
                "home_def_home":         h_def_home,
                "away_att_away":         a_att_away,
                "away_def_away":         a_def_away,
                # v4.2: head-to-head
                "h2h_home_win_rate":     h2h_home_win_rate,
                "h2h_goal_diff":         h2h_goal_diff,
                # v4.3: closing line + corners + referee
                "line_movement_home":    line_movement_home,
                "corners_diff":          corners_diff,
                "referee_home_bias":     ref_bias,
            }

            outcome  = "home" if ftr == "H" else "draw" if ftr == "D" else "away"
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

            # Update stats AFTER building features (no lookahead bias)
            elo.update(home, away, hg, ag)
            stats.update(home, away, hg, ag, h_sot_val, a_sot_val, date,
                         h_corners_val, a_corners_val)
            # Referee stats update
            if ref_name and ref_name.lower() != "nan" and ref_name != "":
                referee_stats[ref_name]["total"] += 1
                if ftr == "H":
                    referee_stats[ref_name]["home_wins"] += 1
            # H2H: persist across seasons, update after match
            h2h_records[pair_key].append({
                "home": home, "away": away, "hg": hg, "ag": ag,
                "date": date.strftime("%Y-%m-%d"),
            })

        season_rows = [r for r in all_rows if r["season"] == year]
        h = sum(1 for r in season_rows if r["actual_outcome"] == "home")
        d = sum(1 for r in season_rows if r["actual_outcome"] == "draw")
        a = sum(1 for r in season_rows if r["actual_outcome"] == "away")
        n = len(season_rows)
        print(f"  Processed {n} matches  H:{h/n*100:.0f}%  D:{d/n*100:.0f}%  A:{a/n*100:.0f}%")

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

    # ── Generate H2H lookup for TypeScript live pipeline ──────────────────────
    h2h_lookup = {}
    for pair, records in h2h_records.items():
        key = "_".join(sorted(pair))  # e.g. "ARS_LIV"
        h2h_lookup[key] = [
            {"home": r["home"], "away": r["away"], "hg": r["hg"], "ag": r["ag"], "date": r["date"]}
            for r in records[-10:]  # keep last 10 meetings
        ]
    h2h_path = ROOT_DIR / "data" / "h2h_lookup.json"
    with open(h2h_path, "w") as f:
        json.dump(h2h_lookup, f, separators=(",", ":"))
    print(f"[OK] H2H lookup written to {h2h_path}  ({len(h2h_lookup)} team pairs)")

    # ── Generate referee bias lookup ──────────────────────────────────────────
    ref_lookup = {}
    for ref_name, stats_dict in referee_stats.items():
        if stats_dict["total"] >= 10:
            bias = (stats_dict["home_wins"] / stats_dict["total"]) - 0.44
            ref_lookup[ref_name] = round(bias, 4)
    ref_path = ROOT_DIR / "data" / "referee_lookup.json"
    with open(ref_path, "w") as f:
        json.dump(ref_lookup, f, indent=2, sort_keys=True)
    print(f"[OK] Referee lookup written to {ref_path}  ({len(ref_lookup)} referees)")

    print(f"\nNext step: python python/train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, default=10,
                        help="Number of most recent seasons to fetch (default: 10)")
    args = parser.parse_args()
    run(num_seasons=min(args.seasons, len(SEASONS)))
