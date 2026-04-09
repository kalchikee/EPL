// EPL Oracle v4.1 — Result Processing
// Fetches completed matches, matches to stored predictions, computes accuracy.

import { logger } from '../logger.js';
import { fetchCompletedResults } from '../api/eplClient.js';
import {
  getPredictionByMatchId, updatePredictionResult, upsertMatchResult, upsertAccuracyLog,
  getPredictionsByGameweek, getSeasonTotals,
} from '../db/database.js';
import type { SeasonTotals } from '../db/database.js';
import { updateElo } from '../features/eloEngine.js';
import { getDominantOutcome } from '../features/marketEdge.js';
import type { Prediction } from '../types.js';

export interface MatchWithResult {
  prediction: Prediction;
  homeScore: number;
  awayScore: number;
}

export interface RecapMetrics {
  accuracy: number;
  brier: number;
  highConvAccuracy: number | null;
  seasonTotals?: SeasonTotals;
}

// Get date range for a gameweek (approximate: 7-day window ending Monday after games)
function getGameweekDateRange(gw: number, season: number): { startDate: string; endDate: string } {
  // EPL GW1 typically starts third weekend of August
  // Approximate: GW1 start = Aug 16 each year, +7 days per gameweek
  const gw1Start = new Date(`${season}-08-09`);
  const weekOffset = (gw - 1) * 7;
  const start = new Date(gw1Start);
  start.setDate(start.getDate() + weekOffset);
  const end = new Date(start);
  end.setDate(end.getDate() + 8); // 8-day window to catch Monday fixtures
  return {
    startDate: start.toISOString().split('T')[0],
    endDate: end.toISOString().split('T')[0],
  };
}

// ─── Process a gameweek's results ─────────────────────────────────────────────

export async function processGameweekResults(
  gameweek: number,
  season: number,
): Promise<{ games: MatchWithResult[]; metrics: RecapMetrics }> {
  const { startDate, endDate } = getGameweekDateRange(gameweek, season);
  logger.info({ gameweek, season, startDate, endDate }, 'Processing gameweek results');

  const results = await fetchCompletedResults(startDate, endDate);
  logger.info({ count: results.length, gameweek, season }, 'Completed results fetched');

  const games: MatchWithResult[] = [];

  for (const result of results) {
    upsertMatchResult(result);

    const pred = getPredictionByMatchId(result.match_id);
    if (!pred) {
      logger.debug({ matchId: result.match_id }, 'No prediction found for match');
      continue;
    }

    // Determine outcome
    const actualOutcome: 'home' | 'draw' | 'away' =
      result.home_score > result.away_score ? 'home'
      : result.home_score < result.away_score ? 'away'
      : 'draw';

    const predictedOutcome = getDominantOutcome(
      pred.calibrated_home_prob,
      pred.calibrated_draw_prob,
      pred.calibrated_away_prob,
    );

    const correct = actualOutcome === predictedOutcome;

    updatePredictionResult(result.match_id, actualOutcome, predictedOutcome, correct);

    // Update Elo after match
    try {
      updateElo(result.home_team, result.away_team, result.home_score, result.away_score);
    } catch (err) {
      logger.debug({ err }, 'Elo update failed');
    }

    games.push({
      prediction: { ...pred, actual_outcome: actualOutcome, predicted_outcome: predictedOutcome, correct },
      homeScore: result.home_score,
      awayScore: result.away_score,
    });
  }

  // If no results found, check if we already have processed predictions
  if (games.length === 0) {
    const gwPreds = getPredictionsByGameweek(gameweek, season).filter(p => p.correct !== undefined);
    if (gwPreds.length > 0) {
      logger.info({ count: gwPreds.length }, 'Using already-processed predictions from DB');
      const metrics = computeMetrics(gwPreds, gameweek, season);
      return { games: [], metrics };
    }
  }

  const metrics = computeMetrics(games.map(g => g.prediction), gameweek, season);
  metrics.seasonTotals = getSeasonTotals(season);

  return { games, metrics };
}

// ─── Compute accuracy metrics ─────────────────────────────────────────────────
// Brier score for multi-class: sum((p_i - outcome_i)^2) for i in {home, draw, away}

function computeMetrics(
  predictions: Prediction[],
  gameweek: number,
  season: number,
): RecapMetrics {
  const evaluated = predictions.filter(p => p.correct !== undefined && p.actual_outcome !== undefined);
  if (evaluated.length === 0) {
    return { accuracy: 0, brier: 0.33, highConvAccuracy: null };
  }

  const correct = evaluated.filter(p => p.correct).length;
  const accuracy = correct / evaluated.length;

  // Multi-class Brier score: mean of sum((p_i - y_i)^2) across 3 outcomes
  let brierSum = 0;
  for (const pred of evaluated) {
    const outcome = pred.actual_outcome!;
    const yH = outcome === 'home' ? 1 : 0;
    const yD = outcome === 'draw' ? 1 : 0;
    const yA = outcome === 'away' ? 1 : 0;
    brierSum += Math.pow(pred.calibrated_home_prob - yH, 2)
      + Math.pow(pred.calibrated_draw_prob - yD, 2)
      + Math.pow(pred.calibrated_away_prob - yA, 2);
  }
  const brier = brierSum / evaluated.length;

  // High-conviction accuracy (max prob ≥ 60%)
  const hc = evaluated.filter(p =>
    Math.max(p.calibrated_home_prob, p.calibrated_draw_prob, p.calibrated_away_prob) >= 0.60
  );
  const highConvAccuracy = hc.length > 0 ? hc.filter(p => p.correct).length / hc.length : null;

  // Log-loss (using predicted outcome probability)
  const logLoss = evaluated.reduce((s, p) => {
    const outcome = p.actual_outcome!;
    const prob = outcome === 'home' ? p.calibrated_home_prob
      : outcome === 'draw' ? p.calibrated_draw_prob
      : p.calibrated_away_prob;
    return s + -Math.log(Math.max(0.001, prob));
  }, 0) / evaluated.length;

  upsertAccuracyLog({
    gameweek,
    season,
    brier_score: brier,
    log_loss: logLoss,
    accuracy,
    high_conf_accuracy: highConvAccuracy ?? 0,
    matches_evaluated: evaluated.length,
  });

  return { accuracy, brier, highConvAccuracy };
}
