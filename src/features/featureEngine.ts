// EPL Oracle v4.1 — Feature Engineering
// Computes 22 features as home-vs-away diffs and absolute values.
// Sources: ESPN standings/stats, Elo ratings, schedule rest.

import { logger } from '../logger.js';
import type { EPLMatch, EPLTeamStats, FeatureVector } from '../types.js';
import { getEloDiff } from './eloEngine.js';
import { getH2HFeatures } from '../api/h2hClient.js';

// ─── Rest days calculation ────────────────────────────────────────────────────

function computeRestDays(matchDate: string, lastMatchDate: string | null): number {
  if (!lastMatchDate) return 7; // assume full week rest for GW1
  const game = new Date(matchDate);
  const last = new Date(lastMatchDate);
  const diffMs = game.getTime() - last.getTime();
  return Math.round(diffMs / (1000 * 60 * 60 * 24));
}

function isShortRest(restDays: number): boolean {
  return restDays <= 3; // ≤3 days = midweek turnaround
}

function hasEuroFatigue(restDays: number): boolean {
  // If played Thursday and now playing Sunday = 3 days (Europa)
  // If played Wednesday and now playing Saturday = 3 days (UCL/EL)
  // Use rest days ≤ 3 as proxy for UEFA midweek fatigue
  return restDays <= 3;
}

// ─── Main feature computation ─────────────────────────────────────────────────

export async function computeFeatures(
  match: EPLMatch,
  allTeamStats: Map<string, EPLTeamStats>,
  homeLastMatchDate: string | null,
  awayLastMatchDate: string | null,
): Promise<FeatureVector> {
  const homeAbbr = match.homeTeam.teamAbbr;
  const awayAbbr = match.awayTeam.teamAbbr;

  logger.debug({ home: homeAbbr, away: awayAbbr }, 'Computing features');

  const homeStats = allTeamStats.get(homeAbbr);
  const awayStats = allTeamStats.get(awayAbbr);

  if (!homeStats || !awayStats) {
    logger.warn({ home: homeAbbr, away: awayAbbr }, 'Missing team stats — using league averages');
  }

  const home = homeStats ?? defaultStats(homeAbbr);
  const away = awayStats ?? defaultStats(awayAbbr);

  // ── Elo ───────────────────────────────────────────────────────────────────
  const eloDiff = getEloDiff(homeAbbr, awayAbbr);

  // ── Attack / Defense ratings ───────────────────────────────────────────────
  // attack_diff > 0 → home scores more per game
  const attackDiff = home.attackRating - away.attackRating;
  // defense_diff > 0 → home concedes less (better defense)
  const defenseDiff = away.defenseRating - home.defenseRating;

  // ── Scoring ───────────────────────────────────────────────────────────────
  const goalsForDiff = home.goalsForPerGame - away.goalsForPerGame;
  const goalsAgainstDiff = home.goalsAgainstPerGame - away.goalsAgainstPerGame;
  const goalDiffDiff = (home.goalsForPerGame - home.goalsAgainstPerGame) - (away.goalsForPerGame - away.goalsAgainstPerGame);
  const netGoalsDiff = goalDiffDiff; // alias

  // ── Form ──────────────────────────────────────────────────────────────────
  const formDiff = home.formLast5 - away.formLast5;
  const homeForm = home.homeFormLast5;
  const awayForm = away.awayFormLast5;

  // ── Table position ─────────────────────────────────────────────────────────
  // Negative diff = home is higher in table (better)
  const positionDiff = home.tablePosition - away.tablePosition;

  // ── Shot quality ───────────────────────────────────────────────────────────
  const shotsOnTargetDiff = home.shotsOnTargetPerGame - away.shotsOnTargetPerGame;

  // ── Possession ────────────────────────────────────────────────────────────
  const possessionDiff = home.possessionPct - away.possessionPct;

  // ── Defensive solidity ─────────────────────────────────────────────────────
  const cleanSheetDiff = home.cleanSheetRate - away.cleanSheetRate;

  // ── Rest / fatigue ────────────────────────────────────────────────────────
  const homeRestDays = computeRestDays(match.matchDate, homeLastMatchDate);
  const awayRestDays = computeRestDays(match.matchDate, awayLastMatchDate);
  const restDaysDiff = homeRestDays - awayRestDays;
  const homeShortRest = isShortRest(homeRestDays) ? 1 : 0;
  const awayShortRest = isShortRest(awayRestDays) ? 1 : 0;
  const homeEuroFatigue = hasEuroFatigue(homeRestDays) ? 1 : 0;
  const awayEuroFatigue = hasEuroFatigue(awayRestDays) ? 1 : 0;

  // ── Venue context ──────────────────────────────────────────────────────────
  const isNeutral = match.venueName.toLowerCase().includes('wembley') ||
    match.venueName.toLowerCase().includes('neutral') ? 1 : 0;

  // ── Home/away split ratings (v4.2) ────────────────────────────────────────
  // Use venue-specific attack/defense for more accurate Poisson lambdas.
  // home team's attack at home vs away team's defense when playing away.
  const homeAttHome = isNeutral ? home.attackRating : home.homeAttackRating;
  const homeDefHome = isNeutral ? home.defenseRating : home.homeDefenseRating;
  const awayAttAway = isNeutral ? away.attackRating : away.awayAttackRating;
  const awayDefAway = isNeutral ? away.defenseRating : away.awayDefenseRating;

  // ── Poisson lambdas: venue-specific (no HOME_ADV multiplier — it's in split stats) ──
  const LEAGUE_AVG_GOALS = 1.35;

  const lambdaHome = Math.max(0.2, Math.min(4.5,
    LEAGUE_AVG_GOALS * homeAttHome * awayDefAway,
  ));
  const lambdaAway = Math.max(0.2, Math.min(4.5,
    LEAGUE_AVG_GOALS * awayAttAway * homeDefHome,
  ));

  logger.debug({
    home: homeAbbr, away: awayAbbr,
    lambdaHome: lambdaHome.toFixed(2), lambdaAway: lambdaAway.toFixed(2),
    eloDiff: eloDiff.toFixed(0),
  }, 'Features computed');

  return {
    elo_diff: eloDiff,
    attack_diff: attackDiff,
    defense_diff: defenseDiff,
    goal_diff_diff: goalDiffDiff,
    goals_for_diff: goalsForDiff,
    goals_against_diff: goalsAgainstDiff,
    net_goals_diff: netGoalsDiff,
    form_diff: formDiff,
    home_form: homeForm,
    away_form: awayForm,
    position_diff: positionDiff,
    shots_on_target_diff: shotsOnTargetDiff,
    possession_diff: possessionDiff,
    clean_sheet_diff: cleanSheetDiff,
    rest_days_diff: restDaysDiff,
    home_short_rest: homeShortRest,
    away_short_rest: awayShortRest,
    home_euro_fatigue: homeEuroFatigue,
    away_euro_fatigue: awayEuroFatigue,
    is_neutral: isNeutral,
    lambda_home: lambdaHome,
    lambda_away: lambdaAway,
    vegas_home_prob: 0,    // filled after odds lookup
    vegas_draw_prob: 0,    // filled after odds lookup
    mc_home_win_prob: 0,   // filled after Monte Carlo
    // Home/away split features (v4.2)
    home_att_home: homeAttHome,
    home_def_home: homeDefHome,
    away_att_away: awayAttAway,
    away_def_away: awayDefAway,
    // Head-to-head (v4.2) — loaded from data/h2h_lookup.json
    ...getH2HFeatures(homeAbbr, awayAbbr),
    // v4.3 features — defaulted at inference time (no live data source yet)
    line_movement_home: 0,    // filled from odds API if closing lines available
    corners_diff: 0,          // no live corners data; neutral default
    referee_home_bias: 0,     // filled from referee_lookup.json if ref is known
  };
}

function defaultStats(abbr: string): EPLTeamStats {
  return {
    teamId: abbr, teamAbbr: abbr, teamName: abbr,
    played: 20, wins: 8, draws: 5, losses: 7, points: 29,
    tablePosition: 10,
    goalsFor: 27, goalsAgainst: 27, goalDifference: 0,
    goalsForPerGame: 1.35, goalsAgainstPerGame: 1.35,
    attackRating: 1.0, defenseRating: 1.0,
    homeAttackRating: 1.18, homeDefenseRating: 0.88,
    awayAttackRating: 0.85, awayDefenseRating: 1.14,
    formLast5: 0.47,
    homeFormLast5: 0.53,
    awayFormLast5: 0.40,
    cleanSheetRate: 0.30,
    bttsRate: 0.55,
    xgFor: 1.35, xgAgainst: 1.35,
    shotsOnTargetPerGame: 3.5,
    possessionPct: 50.0,
  };
}
