// EPL Oracle v4.1 — Elo Rating Engine
// Soccer Elo: K=20, home field advantage ~75 pts, offseason regression to mean.
// Seeded from 2024-25 season final ratings.

import { getElo, upsertElo, getAllElos } from '../db/database.js';

const LEAGUE_MEAN_ELO = 1500;
const K_FACTOR = 20;
const HOME_FIELD_ELO = 75;      // EPL home advantage ~75 Elo points
const OFFSEASON_REGRESSION = 0.70; // how much prior rating carries into next season

// ─── 2024-25 final Elo seeds → carried into 2025-26 ─────────────────────────
// Based on 2024-25 EPL final standings and performance
const ELO_SEEDS_2024_25: Record<string, number> = {
  LIV: 1720,   // Dominant 2024-25 champions
  ARS: 1670,   // Runner-up / top contender
  MCI: 1645,   // Declining from peak, still top 3
  CHE: 1590,   // Improved under new management
  NEW: 1570,   // Strong UCL/UEL qualifier
  AVL: 1555,   // UCL regular
  TOT: 1540,   // Solid mid-table to European
  NFO: 1520,   // Overperformer, solid mid-table
  BHA: 1510,   // Consistent progressive side
  FUL: 1500,   // Solid lower-top half
  MUN: 1490,   // Struggling, rebuilding
  WOL: 1480,   // Battling mid-table
  BRE: 1475,   // Solid lower-mid
  CRY: 1465,   // Lower mid-table
  EVE: 1455,   // Relegation battler, survived
  WHU: 1445,   // Mid-table / lower
  BOU: 1440,   // Solid lower table
  // Promoted 2025-26 teams (lower seeds)
  LEE: 1430,   // Leeds United — Championship winners
  BUR: 1415,   // Burnley — promoted
  SUN: 1405,   // Sunderland — promoted
};

// ─── Seed Elos into DB (idempotent) ──────────────────────────────────────────

export function seedElos(): void {
  const existing = getAllElos();
  if (existing.length >= 18) return; // already seeded (at least 18 of 20 teams present)

  const now = new Date().toISOString();
  for (const [abbr, rating] of Object.entries(ELO_SEEDS_2024_25)) {
    // Regress to mean for new season
    const regressedRating = OFFSEASON_REGRESSION * rating + (1 - OFFSEASON_REGRESSION) * LEAGUE_MEAN_ELO;
    upsertElo({ teamAbbr: abbr, rating: regressedRating, updatedAt: now });
  }
}

// ─── Elo difference ───────────────────────────────────────────────────────────

export function getEloDiff(homeAbbr: string, awayAbbr: string): number {
  return getElo(homeAbbr) - getElo(awayAbbr);
}

// ─── Expected win probability from Elo (includes home field advantage) ───────

export function eloWinProb(homeAbbr: string, awayAbbr: string): number {
  const homeElo = getElo(homeAbbr) + HOME_FIELD_ELO;
  const awayElo = getElo(awayAbbr);
  // Standard Elo formula
  return 1 / (1 + Math.pow(10, (awayElo - homeElo) / 400));
}

// ─── Three-way Elo probability using draw model ───────────────────────────────
// Approximation: draw probability peaks when teams are evenly matched.
// P(draw) ≈ 0.28 * (1 - 2 * |P(home) - 0.5|) + 0.15
// This ensures draws are most likely when win prob is near 0.5.

export function eloThreeWayProb(homeAbbr: string, awayAbbr: string): {
  homeWin: number; draw: number; awayWin: number;
} {
  const pHomeRaw = eloWinProb(homeAbbr, awayAbbr);

  // Draw probability: highest when teams are evenly matched
  const balance = 1 - 2 * Math.abs(pHomeRaw - 0.5);
  const pDraw = 0.15 + 0.20 * balance;

  // Rescale home and away to sum to (1 - pDraw)
  const remaining = 1 - pDraw;
  const pHome = pHomeRaw * remaining;
  const pAway = (1 - pHomeRaw) * remaining;

  return { homeWin: pHome, draw: pDraw, awayWin: pAway };
}

// ─── Elo update after match ───────────────────────────────────────────────────

export function updateElo(
  homeAbbr: string,
  awayAbbr: string,
  homeScore: number,
  awayScore: number,
): void {
  const homeElo = getElo(homeAbbr);
  const awayElo = getElo(awayAbbr);

  const homeExpected = 1 / (1 + Math.pow(10, (awayElo - (homeElo + HOME_FIELD_ELO)) / 400));
  const homeActual = homeScore > awayScore ? 1 : homeScore < awayScore ? 0 : 0.5;

  // Goal margin multiplier for soccer (capped at log(1+4) to avoid overreaction)
  const margin = Math.abs(homeScore - awayScore);
  const movMultiplier = margin === 0 ? 0.5 : Math.log(1 + Math.min(margin, 4)) / Math.log(5);

  const homeChange = K_FACTOR * movMultiplier * (homeActual - homeExpected);

  const now = new Date().toISOString();
  upsertElo({ teamAbbr: homeAbbr, rating: homeElo + homeChange, updatedAt: now });
  upsertElo({ teamAbbr: awayAbbr, rating: awayElo - homeChange, updatedAt: now });
}
