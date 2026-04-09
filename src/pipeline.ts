// EPL Oracle v4.1 — Gameweek Pipeline
// Orchestrates: Fetch → Features → Poisson MC → ML Model → Edge → Store → Print

import { logger } from './logger.js';
import { fetchGameweekSchedule, fetchAllTeamStats, getCurrentGameweekInfo } from './api/eplClient.js';
import { computeFeatures } from './features/featureEngine.js';
import { runPoisson } from './models/monteCarlo.js';
import { loadModel, predict as mlPredict, isModelLoaded, getModelInfo } from './models/metaModel.js';
import { upsertPrediction, initDb, getPredictionsByGameweek } from './db/database.js';
import { getOddsForMatch, loadOddsApiLines } from './api/oddsClient.js';
import { loadH2H } from './api/h2hClient.js';
import { fetchInjuries, applyInjuryAdjustment, getTeamInjuries } from './api/injuryClient.js';
import { seedElos } from './features/eloEngine.js';
import { computeEdge, getConfidenceTier } from './features/marketEdge.js';
import type { EPLMatch, Prediction, PipelineOptions } from './types.js';

const MODEL_VERSION = '4.2.0';

// ─── Last match date lookup ───────────────────────────────────────────────────

async function getLastMatchDates(
  gw: number,
  season: number,
  homeAbbr: string,
  awayAbbr: string,
): Promise<{ homeLastMatch: string | null; awayLastMatch: string | null }> {
  if (gw <= 1) return { homeLastMatch: null, awayLastMatch: null };
  try {
    const prevGwMatches = await fetchGameweekSchedule(gw - 1, season);
    const homeMatch = prevGwMatches.find(
      m => m.homeTeam.teamAbbr === homeAbbr || m.awayTeam.teamAbbr === homeAbbr,
    );
    const awayMatch = prevGwMatches.find(
      m => m.homeTeam.teamAbbr === awayAbbr || m.awayTeam.teamAbbr === awayAbbr,
    );
    return {
      homeLastMatch: homeMatch?.matchDate ?? null,
      awayLastMatch: awayMatch?.matchDate ?? null,
    };
  } catch {
    return { homeLastMatch: null, awayLastMatch: null };
  }
}

// ─── Main pipeline ────────────────────────────────────────────────────────────

export async function runPipeline(options: PipelineOptions = {}): Promise<Prediction[]> {
  let { gameweek, season } = options;
  if (!gameweek || !season) {
    const current = await getCurrentGameweekInfo();
    gameweek = options.gameweek ?? current.gameweek;
    season = options.season ?? current.season;
  }

  logger.info({ gameweek, season, version: MODEL_VERSION }, '=== EPL Oracle v4.2 Pipeline Start ===');

  await initDb();
  seedElos();

  const modelLoaded = loadModel();
  if (modelLoaded) {
    const info = getModelInfo();
    logger.info({ version: info?.version, brier: info?.avg_brier }, 'ML model active');
  } else {
    logger.info('ML model not found — using Poisson MC only. Run: npm run train');
  }

  // Load H2H lookup (precomputed from historical data)
  loadH2H();

  await loadOddsApiLines();

  // Load injury data (optional — requires API_FOOTBALL_KEY)
  await fetchInjuries(season);

  const teamStats = await fetchAllTeamStats(season);
  const matches = await fetchGameweekSchedule(gameweek, season);

  if (matches.length === 0) {
    logger.warn({ gameweek, season }, 'No matches found for this gameweek');
    return [];
  }

  logger.info({ gameweek, season, matches: matches.length }, 'Gameweek schedule fetched');

  const predictions: Prediction[] = [];

  for (const match of matches) {
    if (match.status.includes('FINAL') || match.status.includes('final')) {
      logger.debug({ matchId: match.matchId, status: match.status }, 'Skipping completed match');
      continue;
    }

    try {
      const pred = await processMatch(match, gameweek, season, teamStats, modelLoaded);
      if (pred) predictions.push(pred);
    } catch (err) {
      logger.error(
        { err, matchId: match.matchId, home: match.homeTeam.teamAbbr, away: match.awayTeam.teamAbbr },
        'Failed to process match',
      );
    }
  }

  logger.info({ processed: predictions.length, gameweek, season }, 'Pipeline complete');

  if (options.verbose !== false) {
    printPredictions(predictions, gameweek, season, modelLoaded);
  }

  return predictions;
}

// ─── Single match processing ──────────────────────────────────────────────────

async function processMatch(
  match: EPLMatch,
  gameweek: number,
  season: number,
  teamStats: Awaited<ReturnType<typeof fetchAllTeamStats>>,
  modelLoaded: boolean,
): Promise<Prediction | null> {
  const homeAbbr = match.homeTeam.teamAbbr;
  const awayAbbr = match.awayTeam.teamAbbr;

  logger.info({ matchId: match.matchId, matchup: `${homeAbbr} vs ${awayAbbr}` }, 'Processing match');

  const { homeLastMatch, awayLastMatch } = await getLastMatchDates(gameweek, season, homeAbbr, awayAbbr);

  const features = await computeFeatures(match, teamStats, homeLastMatch, awayLastMatch);

  const poisson = runPoisson(features);
  features.mc_home_win_prob = poisson.home_win_prob;
  features.lambda_home = poisson.expected_home_goals;
  features.lambda_away = poisson.expected_away_goals;

  // Vegas odds
  let vegas_home_prob: number | undefined;
  let vegas_draw_prob: number | undefined;
  let vegas_away_prob: number | undefined;
  let edge_home: number | undefined;
  let edge_draw: number | undefined;
  let edge_away: number | undefined;

  const matchOdds = getOddsForMatch(
    homeAbbr, awayAbbr,
    match.homeOdds, match.drawOdds, match.awayOdds,
    match.homeMoneyLine, match.drawMoneyLine, match.awayMoneyLine,
  );

  if (matchOdds) {
    vegas_home_prob = matchOdds.homeImpliedProb;
    vegas_draw_prob = matchOdds.drawImpliedProb;
    vegas_away_prob = matchOdds.awayImpliedProb;
    features.vegas_home_prob = vegas_home_prob;
    features.vegas_draw_prob = vegas_draw_prob;
  }

  // ML calibration
  let calibrated_home_prob: number;
  let calibrated_draw_prob: number;
  let calibrated_away_prob: number;

  if (modelLoaded && isModelLoaded()) {
    const calibrated = mlPredict(features, poisson.home_win_prob, poisson.draw_prob, poisson.away_win_prob);
    calibrated_home_prob = calibrated.home;
    calibrated_draw_prob = calibrated.draw;
    calibrated_away_prob = calibrated.away;
  } else {
    calibrated_home_prob = poisson.home_win_prob;
    calibrated_draw_prob = poisson.draw_prob;
    calibrated_away_prob = poisson.away_win_prob;
  }

  // Apply injury adjustment (post-model, ±4% max based on squad fitness)
  const injuryAdjusted = applyInjuryAdjustment(homeAbbr, awayAbbr, {
    home: calibrated_home_prob,
    draw: calibrated_draw_prob,
    away: calibrated_away_prob,
  });
  calibrated_home_prob = injuryAdjusted.home;
  calibrated_draw_prob = injuryAdjusted.draw;
  calibrated_away_prob = injuryAdjusted.away;

  // Edge detection
  if (vegas_home_prob !== undefined && vegas_draw_prob !== undefined && vegas_away_prob !== undefined) {
    edge_home = computeEdge(calibrated_home_prob, vegas_home_prob).edge;
    edge_draw = computeEdge(calibrated_draw_prob, vegas_draw_prob).edge;
    edge_away = computeEdge(calibrated_away_prob, vegas_away_prob).edge;
  }

  const prediction: Prediction = {
    match_date: match.matchDate,
    match_id: match.matchId,
    gameweek,
    season,
    home_team: homeAbbr,
    away_team: awayAbbr,
    venue: match.venueName,
    feature_vector: features,
    home_win_prob: poisson.home_win_prob,
    draw_prob: poisson.draw_prob,
    away_win_prob: poisson.away_win_prob,
    calibrated_home_prob,
    calibrated_draw_prob,
    calibrated_away_prob,
    vegas_home_prob,
    vegas_draw_prob,
    vegas_away_prob,
    edge_home,
    edge_draw,
    edge_away,
    model_version: MODEL_VERSION,
    expected_home_goals: poisson.expected_home_goals,
    expected_away_goals: poisson.expected_away_goals,
    most_likely_score: `${poisson.most_likely_score[0]}-${poisson.most_likely_score[1]}`,
    over_2_5_prob: poisson.over_2_5_prob,
    btts_prob: poisson.btts_prob,
    created_at: new Date().toISOString(),
  };

  upsertPrediction(prediction);
  return prediction;
}

// ─── Console output ───────────────────────────────────────────────────────────

function printPredictions(
  predictions: Prediction[],
  gameweek: number,
  season: number,
  mlActive = false,
): void {
  if (predictions.length === 0) {
    console.log(`\nNo predictions for GW${gameweek}, ${season}-${(season + 1).toString().slice(2)}\n`);
    return;
  }

  const label = mlActive ? 'ML+Isotonic' : 'Poisson MC';
  const width = 115;

  console.log('\n' + '═'.repeat(width));
  console.log(`  EPL ORACLE v4.1  ·  GW${gameweek} ${season}-${(season + 1).toString().slice(2)}  ·  ${predictions.length} matches  ·  [${label}]`);
  console.log('═'.repeat(width));

  const sorted = [...predictions].sort((a, b) => {
    const maxA = Math.max(a.calibrated_home_prob, a.calibrated_away_prob);
    const maxB = Math.max(b.calibrated_home_prob, b.calibrated_away_prob);
    return maxB - maxA;
  });

  console.log('\n' + [
    pad('MATCHUP', 28), pad('H WIN%', 8), pad('DRAW%', 7), pad('A WIN%', 8),
    pad('PROJ', 9), pad('O2.5', 6), pad('BTTS', 6), pad('EDGE-H', 8), 'PICK',
  ].join('  '));
  console.log('─'.repeat(width));

  for (const p of sorted) {
    const matchup = `${p.home_team} vs ${p.away_team}`;
    const hPct = (p.calibrated_home_prob * 100).toFixed(1) + '%';
    const dPct = (p.calibrated_draw_prob * 100).toFixed(1) + '%';
    const aPct = (p.calibrated_away_prob * 100).toFixed(1) + '%';
    const edgeHStr = p.edge_home !== undefined
      ? (p.edge_home >= 0 ? '+' : '') + (p.edge_home * 100).toFixed(1) + '%'
      : '—';
    const o25 = (p.over_2_5_prob * 100).toFixed(0) + '%';
    const btts = (p.btts_prob * 100).toFixed(0) + '%';

    const maxProb = Math.max(p.calibrated_home_prob, p.calibrated_draw_prob, p.calibrated_away_prob);
    const pick = maxProb === p.calibrated_home_prob ? p.home_team
      : maxProb === p.calibrated_away_prob ? p.away_team : 'DRAW';
    const tier = getConfidenceTier(p.calibrated_home_prob, p.calibrated_draw_prob, p.calibrated_away_prob);
    const marker = tier === 'extreme' || tier === 'high_conviction' ? ' ★' : '';

    console.log([
      pad(matchup, 28), pad(hPct, 8), pad(dPct, 7), pad(aPct, 8),
      pad(p.most_likely_score, 9), pad(o25, 6), pad(btts, 6),
      pad(edgeHStr, 8), pick + marker,
    ].join('  '));
  }

  console.log('─'.repeat(width));
  console.log('★ = high conviction  |  EDGE-H = home edge vs vig-removed market  |  EPL Oracle v4.1\n');
}

function pad(s: string, w: number): string {
  return s.length >= w ? s.slice(0, w) : s + ' '.repeat(w - s.length);
}
