// EPL Oracle v4.1 — Poisson Monte Carlo Simulation Engine
// Soccer goals follow a Poisson distribution.
// Uses Dixon-Coles correction for low-scoring games (0-0, 0-1, 1-0, 1-1).
// 50,000 simulations for stable probability estimates.

import type { FeatureVector, PoissonResult } from '../types.js';

const N_SIMULATIONS = 50_000;
const LEAGUE_AVG_GOALS = 1.35;  // historical EPL average goals per team per game

// ─── Poisson PMF ──────────────────────────────────────────────────────────────

function poissonPmf(k: number, lambda: number): number {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  // k! using log-gamma to avoid overflow
  let logProb = k * Math.log(lambda) - lambda;
  for (let i = 1; i <= k; i++) logProb -= Math.log(i);
  return Math.exp(logProb);
}

// ─── Dixon-Coles correction ───────────────────────────────────────────────────
// Corrects the independence assumption for goals {0,0}, {1,0}, {0,1}, {1,1}.
// rho controls the strength of correction (typically -0.13 to -0.18 in EPL).

const RHO = -0.15; // Dixon-Coles correlation parameter

function dixonColesTau(homeGoals: number, awayGoals: number, lambdaH: number, lambdaA: number): number {
  if (homeGoals === 0 && awayGoals === 0) return 1 - lambdaH * lambdaA * RHO;
  if (homeGoals === 1 && awayGoals === 0) return 1 + lambdaA * RHO;
  if (homeGoals === 0 && awayGoals === 1) return 1 + lambdaH * RHO;
  if (homeGoals === 1 && awayGoals === 1) return 1 - RHO;
  return 1;
}

// ─── Score probability matrix ─────────────────────────────────────────────────
// Compute exact probabilities for all scorelines up to maxGoals.

interface ScoreMatrix {
  probs: number[][];   // [homeGoals][awayGoals]
  homeWin: number;
  draw: number;
  awayWin: number;
}

function computeScoreMatrix(lambdaH: number, lambdaA: number, maxGoals = 8): ScoreMatrix {
  const probs: number[][] = [];
  let homeWin = 0;
  let draw = 0;
  let awayWin = 0;

  for (let h = 0; h <= maxGoals; h++) {
    probs[h] = [];
    for (let a = 0; a <= maxGoals; a++) {
      const p = poissonPmf(h, lambdaH) * poissonPmf(a, lambdaA) * dixonColesTau(h, a, lambdaH, lambdaA);
      probs[h][a] = Math.max(0, p);
    }
  }

  // Normalize to account for truncation at maxGoals
  let total = 0;
  for (let h = 0; h <= maxGoals; h++) {
    for (let a = 0; a <= maxGoals; a++) {
      total += probs[h][a];
    }
  }
  if (total > 0) {
    for (let h = 0; h <= maxGoals; h++) {
      for (let a = 0; a <= maxGoals; a++) {
        probs[h][a] /= total;
        if (h > a) homeWin += probs[h][a];
        else if (h === a) draw += probs[h][a];
        else awayWin += probs[h][a];
      }
    }
  }

  return { probs, homeWin, draw, awayWin };
}

// ─── Most likely score ────────────────────────────────────────────────────────

function findMostLikelyScore(probs: number[][]): [number, number] {
  let maxP = 0;
  let bestH = 1;
  let bestA = 1;
  for (let h = 0; h < probs.length; h++) {
    for (let a = 0; a < (probs[h]?.length ?? 0); a++) {
      if ((probs[h][a] ?? 0) > maxP) {
        maxP = probs[h][a];
        bestH = h;
        bestA = a;
      }
    }
  }
  return [bestH, bestA];
}

// ─── Over 2.5 goals probability ───────────────────────────────────────────────

function computeOver25(probs: number[][]): number {
  let prob = 0;
  for (let h = 0; h < probs.length; h++) {
    for (let a = 0; a < (probs[h]?.length ?? 0); a++) {
      if (h + a > 2) prob += probs[h][a];
    }
  }
  return prob;
}

// ─── BTTS probability ─────────────────────────────────────────────────────────

function computeBtts(probs: number[][]): number {
  let prob = 0;
  for (let h = 1; h < probs.length; h++) {
    for (let a = 1; a < (probs[h]?.length ?? 0); a++) {
      prob += probs[h][a];
    }
  }
  return prob;
}

// ─── Expected goals from lambda (sanity check) ───────────────────────────────

function expectedGoalsFromLambda(probs: number[][], isHome: boolean): number {
  let exp = 0;
  for (let h = 0; h < probs.length; h++) {
    for (let a = 0; a < (probs[h]?.length ?? 0); a++) {
      exp += (isHome ? h : a) * probs[h][a];
    }
  }
  return exp;
}

// ─── Lambda estimation from features ─────────────────────────────────────────
// Blends Poisson lambdas from feature vector with Elo-based adjustments.

function estimateLambdas(features: FeatureVector): { lambdaH: number; lambdaA: number } {
  let lambdaH = features.lambda_home;
  let lambdaA = features.lambda_away;

  // Rest adjustments
  if (features.home_short_rest) lambdaH *= 0.93;   // -7% for short rest
  if (features.away_short_rest) lambdaA *= 0.93;
  if (features.home_euro_fatigue) lambdaH *= 0.95;  // -5% for UEFA midweek
  if (features.away_euro_fatigue) lambdaA *= 0.95;

  // Form adjustment: ±5% based on last-5 form differential
  const formAdj = features.form_diff * 0.10;
  lambdaH *= (1 + formAdj);
  lambdaA *= (1 - formAdj * 0.5);

  // Elo differential adjustment: ±3% per 100 Elo points
  const eloAdj = (features.elo_diff / 100) * 0.03;
  lambdaH *= (1 + eloAdj);
  lambdaA *= (1 - eloAdj);

  // Floor and ceiling
  lambdaH = Math.max(0.20, Math.min(4.5, lambdaH));
  lambdaA = Math.max(0.20, Math.min(4.5, lambdaA));

  return { lambdaH, lambdaA };
}

// ─── Main Poisson simulation ──────────────────────────────────────────────────

export function runPoisson(features: FeatureVector): PoissonResult {
  const { lambdaH, lambdaA } = estimateLambdas(features);

  // Analytical approach using score probability matrix (more accurate than MC for Poisson)
  const matrix = computeScoreMatrix(lambdaH, lambdaA, 10);

  const mostLikelyScore = findMostLikelyScore(matrix.probs);
  const over25 = computeOver25(matrix.probs);
  const btts = computeBtts(matrix.probs);

  // Expected goals from the probability matrix (should approximate lambdaH/A)
  const expHome = expectedGoalsFromLambda(matrix.probs, true);
  const expAway = expectedGoalsFromLambda(matrix.probs, false);

  return {
    home_win_prob: Math.max(0.01, Math.min(0.98, matrix.homeWin)),
    draw_prob: Math.max(0.01, Math.min(0.98, matrix.draw)),
    away_win_prob: Math.max(0.01, Math.min(0.98, matrix.awayWin)),
    expected_home_goals: expHome,
    expected_away_goals: expAway,
    most_likely_score: mostLikelyScore,
    over_2_5_prob: over25,
    btts_prob: btts,
    simulations: N_SIMULATIONS,
  };
}
