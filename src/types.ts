// EPL Oracle v4.1 — Core Type Definitions

// ─── EPL team / match types ───────────────────────────────────────────────────

export interface EPLTeamStats {
  teamId: string;
  teamAbbr: string;
  teamName: string;
  // Table
  played: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  tablePosition: number;
  // Scoring
  goalsFor: number;
  goalsAgainst: number;
  goalDifference: number;
  goalsForPerGame: number;
  goalsAgainstPerGame: number;
  // Efficiency (attack/defense ratings for Poisson model)
  attackRating: number;        // relative to league avg (1.0 = avg)
  defenseRating: number;       // relative to league avg (1.0 = avg), lower = better
  // Home/away specific attack/defense (v4.2)
  homeAttackRating: number;    // attack rating from home games only
  homeDefenseRating: number;   // defense rating (GA) from home games only
  awayAttackRating: number;    // attack rating from away games only
  awayDefenseRating: number;   // defense rating (GA) from away games only
  // Form (last 5 matches: W=3, D=1, L=0 → normalized to 0-1)
  formLast5: number;           // 0.0–1.0
  homeFormLast5: number;       // home-specific form
  awayFormLast5: number;       // away-specific form
  // Advanced
  cleanSheetRate: number;      // fraction of games with clean sheet
  bttsRate: number;            // both teams to score rate
  xgFor: number;               // expected goals for per game (if available)
  xgAgainst: number;           // expected goals against per game
  shotsOnTargetPerGame: number;
  possessionPct: number;       // average possession percentage
}

export interface EPLMatch {
  matchId: string;
  matchDate: string;          // YYYY-MM-DD
  matchTime: string;          // ISO datetime UTC
  gameweek: number;
  season: number;             // e.g. 2025 = 2025-26 season
  status: string;             // 'STATUS_SCHEDULED' | 'STATUS_IN_PROGRESS' | 'STATUS_FINAL'
  homeTeam: EPLMatchTeam;
  awayTeam: EPLMatchTeam;
  venueName: string;
  venueCity: string;
  // Embedded odds (if available)
  homeOdds?: number;          // decimal odds (e.g. 2.10)
  drawOdds?: number;
  awayOdds?: number;
  homeMoneyLine?: number;     // American odds
  awayMoneyLine?: number;
  drawMoneyLine?: number;
  vegasTotal?: number;        // over/under goals
}

export interface EPLMatchTeam {
  teamId: string;
  teamAbbr: string;
  teamName: string;
  score?: number;
}

// ─── Feature vector ───────────────────────────────────────────────────────────

export interface FeatureVector {
  // Team strength
  elo_diff: number;              // home Elo - away Elo
  attack_diff: number;           // home attack rating - away attack rating
  defense_diff: number;          // home defense rating - away defense rating (positive = home better def)
  goal_diff_diff: number;        // home GD/game - away GD/game

  // Scoring
  goals_for_diff: number;        // home goals/game - away goals/game
  goals_against_diff: number;    // home GA/game - away GA/game
  net_goals_diff: number;        // (gf - ga) home - away

  // Form
  form_diff: number;             // home form - away form (last 5)
  home_form: number;             // home team's home-specific form
  away_form: number;             // away team's away-specific form

  // Table position
  position_diff: number;         // home table pos - away table pos (negative = home higher)

  // Shot quality
  shots_on_target_diff: number;

  // Possession
  possession_diff: number;       // home possession% - away possession%

  // Defensive solidity
  clean_sheet_diff: number;      // home clean sheet rate - away clean sheet rate

  // Fatigue / schedule
  rest_days_diff: number;        // home days since last match - away days since last match
  home_short_rest: number;       // 1 if home played ≤ 3 days ago
  away_short_rest: number;       // 1 if away played ≤ 3 days ago
  home_euro_fatigue: number;     // 1 if home played UEFA mid-week
  away_euro_fatigue: number;     // 1 if away played UEFA mid-week

  // Context
  is_neutral: number;            // 1 if neutral site (rare)

  // Poisson model lambdas (set at prediction time)
  lambda_home: number;           // expected home goals
  lambda_away: number;           // expected away goals

  // Vegas
  vegas_home_prob: number;       // vig-removed market home win probability
  vegas_draw_prob: number;       // vig-removed draw probability
  mc_home_win_prob: number;      // Monte Carlo home win probability (set at prediction time)

  // Home/away split attack-defense ratings (v4.2)
  home_att_home: number;         // home team attack rating from home games only
  home_def_home: number;         // home team defense rating (GA) from home games only
  away_att_away: number;         // away team attack rating from away games only
  away_def_away: number;         // away team defense rating (GA) from away games only

  // Head-to-head history (v4.2)
  h2h_home_win_rate: number;     // fraction of last 5 meetings where home team won
  h2h_goal_diff: number;         // average (home goals - away goals) in last 5 meetings
}

// ─── Model outputs ────────────────────────────────────────────────────────────

export interface PoissonResult {
  home_win_prob: number;
  draw_prob: number;
  away_win_prob: number;
  expected_home_goals: number;
  expected_away_goals: number;
  most_likely_score: [number, number];    // [home, away]
  over_2_5_prob: number;                  // probability over 2.5 total goals
  btts_prob: number;                      // both teams to score probability
  simulations: number;
}

export interface Prediction {
  match_date: string;
  match_id: string;
  gameweek: number;
  season: number;
  home_team: string;
  away_team: string;
  venue: string;
  feature_vector: FeatureVector;
  home_win_prob: number;
  draw_prob: number;
  away_win_prob: number;
  calibrated_home_prob: number;
  calibrated_draw_prob: number;
  calibrated_away_prob: number;
  vegas_home_prob?: number;
  vegas_draw_prob?: number;
  vegas_away_prob?: number;
  edge_home?: number;
  edge_draw?: number;
  edge_away?: number;
  model_version: string;
  expected_home_goals: number;
  expected_away_goals: number;
  most_likely_score: string;
  over_2_5_prob: number;
  btts_prob: number;
  actual_outcome?: 'home' | 'draw' | 'away';
  predicted_outcome?: 'home' | 'draw' | 'away';
  correct?: boolean;
  created_at: string;
}

export interface EloRating {
  teamAbbr: string;
  rating: number;
  updatedAt: string;
}

export interface AccuracyLog {
  gameweek: number;
  season: number;
  brier_score: number;
  log_loss: number;
  accuracy: number;
  high_conf_accuracy: number;
  matches_evaluated: number;
}

export interface MatchResult {
  match_id: string;
  date: string;
  gameweek: number;
  season: number;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  venue: string;
}

export interface PipelineOptions {
  gameweek?: number;
  season?: number;
  forceRefresh?: boolean;
  verbose?: boolean;
}

// ─── Edge detection ───────────────────────────────────────────────────────────

export type EdgeCategory = 'none' | 'small' | 'meaningful' | 'large' | 'extreme';

export interface EdgeResult {
  modelProb: number;
  vegasProb: number;
  edge: number;
  edgeCategory: EdgeCategory;
}

// ─── Season helpers ───────────────────────────────────────────────────────────

export interface EPLGameweekInfo {
  season: number;
  gameweek: number;
  startDate: string;    // YYYY-MM-DD
  endDate: string;
}
