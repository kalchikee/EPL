// EPL Oracle v4.1 — SQLite Database Layer (sql.js — pure JS, no native build)

import initSqlJs, { type Database as SqlJsDatabase } from 'sql.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { Prediction, MatchResult, AccuracyLog, EloRating } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DB_PATH = resolve(
  process.env.DB_PATH
    ? process.env.DB_PATH.startsWith('.')
      ? resolve(__dirname, '../../', process.env.DB_PATH)
      : process.env.DB_PATH
    : resolve(__dirname, '../../data/epl_oracle.db'),
);

mkdirSync(dirname(DB_PATH), { recursive: true });

let _db: SqlJsDatabase | null = null;
let _SQL: Awaited<ReturnType<typeof initSqlJs>> | null = null;

// ─── Initialization ───────────────────────────────────────────────────────────

export async function initDb(): Promise<SqlJsDatabase> {
  if (_db) return _db;
  _SQL = await initSqlJs();
  if (existsSync(DB_PATH)) {
    const fileBuffer = readFileSync(DB_PATH);
    _db = new _SQL.Database(fileBuffer);
  } else {
    _db = new _SQL.Database();
  }
  initializeSchema(_db);
  persistDb();
  return _db;
}

export function getDb(): SqlJsDatabase {
  if (!_db) throw new Error('Database not initialized. Call initDb() first.');
  return _db;
}

export function persistDb(): void {
  if (!_db) return;
  const data = _db.export();
  writeFileSync(DB_PATH, Buffer.from(data));
}

function run(sql: string, params: (string | number | null | undefined)[] = []): void {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.run(params.map(p => (p === undefined ? null : p)));
  stmt.free();
  persistDb();
}

function queryAll<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T[] {
  const db = getDb();
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const results: T[] = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject() as T);
  }
  stmt.free();
  return results;
}

function queryOne<T = Record<string, unknown>>(sql: string, params: (string | number | null)[] = []): T | undefined {
  return queryAll<T>(sql, params)[0];
}

// ─── Schema ───────────────────────────────────────────────────────────────────

function initializeSchema(db: SqlJsDatabase): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS elo_ratings (
      team_abbr TEXT PRIMARY KEY,
      rating REAL NOT NULL DEFAULT 1500,
      games_played INTEGER NOT NULL DEFAULT 0,
      season INTEGER NOT NULL DEFAULT 2025,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      match_id TEXT NOT NULL,
      match_date TEXT NOT NULL,
      gameweek INTEGER NOT NULL DEFAULT 0,
      season INTEGER NOT NULL DEFAULT 2025,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      venue TEXT NOT NULL DEFAULT '',
      feature_vector TEXT NOT NULL,
      home_win_prob REAL NOT NULL,
      draw_prob REAL NOT NULL,
      away_win_prob REAL NOT NULL,
      calibrated_home_prob REAL NOT NULL,
      calibrated_draw_prob REAL NOT NULL,
      calibrated_away_prob REAL NOT NULL,
      vegas_home_prob REAL,
      vegas_draw_prob REAL,
      vegas_away_prob REAL,
      edge_home REAL,
      edge_draw REAL,
      edge_away REAL,
      expected_home_goals REAL NOT NULL DEFAULT 0,
      expected_away_goals REAL NOT NULL DEFAULT 0,
      most_likely_score TEXT NOT NULL DEFAULT '',
      over_2_5_prob REAL NOT NULL DEFAULT 0,
      btts_prob REAL NOT NULL DEFAULT 0,
      model_version TEXT NOT NULL DEFAULT '4.1.0',
      actual_outcome TEXT,
      predicted_outcome TEXT,
      correct INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS accuracy_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      gameweek INTEGER NOT NULL,
      season INTEGER NOT NULL,
      brier_score REAL NOT NULL DEFAULT 0,
      log_loss REAL NOT NULL DEFAULT 0,
      accuracy REAL NOT NULL DEFAULT 0,
      high_conf_accuracy REAL NOT NULL DEFAULT 0,
      matches_evaluated INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(gameweek, season)
    );

    CREATE TABLE IF NOT EXISTS match_results (
      match_id TEXT PRIMARY KEY,
      date TEXT NOT NULL,
      gameweek INTEGER NOT NULL DEFAULT 0,
      season INTEGER NOT NULL DEFAULT 2025,
      home_team TEXT NOT NULL,
      away_team TEXT NOT NULL,
      home_score INTEGER NOT NULL,
      away_score INTEGER NOT NULL,
      venue TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS model_registry (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      version TEXT NOT NULL UNIQUE,
      train_seasons TEXT NOT NULL DEFAULT '',
      test_brier REAL NOT NULL DEFAULT 0,
      test_accuracy REAL NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);

  const cnt = queryOne<{ cnt: number }>('SELECT COUNT(*) as cnt FROM model_registry');
  if (!cnt || cnt.cnt === 0) {
    db.run(
      `INSERT OR IGNORE INTO model_registry (version, train_seasons, test_brier, test_accuracy)
       VALUES (?, ?, ?, ?)`,
      ['4.1.0', '2020-2025', 0, 0],
    );
  }
}

// ─── Elo helpers ──────────────────────────────────────────────────────────────

export function upsertElo(rating: EloRating): void {
  run(
    `INSERT INTO elo_ratings (team_abbr, rating, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT(team_abbr) DO UPDATE SET
       rating = excluded.rating,
       updated_at = excluded.updated_at`,
    [rating.teamAbbr, rating.rating, rating.updatedAt],
  );
}

export function getElo(teamAbbr: string): number {
  const row = queryOne<{ rating: number }>(
    'SELECT rating FROM elo_ratings WHERE team_abbr = ?',
    [teamAbbr],
  );
  return row?.rating ?? 1500;
}

export function getAllElos(): EloRating[] {
  return queryAll<{ team_abbr: string; rating: number; updated_at: string }>(
    'SELECT team_abbr, rating, updated_at FROM elo_ratings ORDER BY rating DESC',
  ).map(r => ({ teamAbbr: r.team_abbr, rating: r.rating, updatedAt: r.updated_at }));
}

// ─── Prediction helpers ───────────────────────────────────────────────────────

export function upsertPrediction(pred: Prediction): void {
  run(`DELETE FROM predictions WHERE match_id = ? AND model_version = ?`, [pred.match_id, pred.model_version]);
  run(
    `INSERT INTO predictions (
       match_id, match_date, gameweek, season, home_team, away_team, venue,
       feature_vector,
       home_win_prob, draw_prob, away_win_prob,
       calibrated_home_prob, calibrated_draw_prob, calibrated_away_prob,
       vegas_home_prob, vegas_draw_prob, vegas_away_prob,
       edge_home, edge_draw, edge_away,
       expected_home_goals, expected_away_goals,
       most_likely_score, over_2_5_prob, btts_prob,
       model_version, created_at
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      pred.match_id, pred.match_date, pred.gameweek, pred.season,
      pred.home_team, pred.away_team, pred.venue,
      JSON.stringify(pred.feature_vector),
      pred.home_win_prob, pred.draw_prob, pred.away_win_prob,
      pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob,
      pred.vegas_home_prob ?? null, pred.vegas_draw_prob ?? null, pred.vegas_away_prob ?? null,
      pred.edge_home ?? null, pred.edge_draw ?? null, pred.edge_away ?? null,
      pred.expected_home_goals, pred.expected_away_goals,
      pred.most_likely_score, pred.over_2_5_prob, pred.btts_prob,
      pred.model_version, pred.created_at,
    ],
  );
}

export function getPredictionsByGameweek(gameweek: number, season: number): Prediction[] {
  const rows = queryAll<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE gameweek = ? AND season = ? ORDER BY calibrated_home_prob DESC',
    [gameweek, season],
  );
  return rows.map(row => ({
    ...row,
    feature_vector: JSON.parse(row.feature_vector as string),
  })) as Prediction[];
}

export function getPredictionByMatchId(matchId: string): Prediction | undefined {
  const row = queryOne<Record<string, unknown>>(
    'SELECT * FROM predictions WHERE match_id = ? ORDER BY created_at DESC LIMIT 1',
    [matchId],
  );
  if (!row) return undefined;
  return { ...row, feature_vector: JSON.parse(row.feature_vector as string) } as Prediction;
}

export function updatePredictionResult(
  matchId: string,
  outcome: 'home' | 'draw' | 'away',
  predictedOutcome: 'home' | 'draw' | 'away',
  correct: boolean,
): void {
  run(
    `UPDATE predictions SET actual_outcome = ?, predicted_outcome = ?, correct = ? WHERE match_id = ?`,
    [outcome, predictedOutcome, correct ? 1 : 0, matchId],
  );
}

// ─── Match result helpers ──────────────────────────────────────────────────────

export function upsertMatchResult(result: MatchResult): void {
  run(
    `INSERT INTO match_results (match_id, date, gameweek, season, home_team, away_team, home_score, away_score, venue)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(match_id) DO UPDATE SET
       home_score = excluded.home_score,
       away_score = excluded.away_score`,
    [
      result.match_id, result.date, result.gameweek, result.season,
      result.home_team, result.away_team,
      result.home_score, result.away_score, result.venue,
    ],
  );
}

// ─── Accuracy helpers ─────────────────────────────────────────────────────────

export function upsertAccuracyLog(log: AccuracyLog): void {
  run(
    `INSERT INTO accuracy_log (gameweek, season, brier_score, log_loss, accuracy, high_conf_accuracy, matches_evaluated)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(gameweek, season) DO UPDATE SET
       brier_score = excluded.brier_score,
       accuracy = excluded.accuracy,
       high_conf_accuracy = excluded.high_conf_accuracy,
       matches_evaluated = excluded.matches_evaluated`,
    [log.gameweek, log.season, log.brier_score, log.log_loss, log.accuracy, log.high_conf_accuracy, log.matches_evaluated],
  );
}

export interface SeasonTotals {
  season: number;
  totalMatches: number;
  totalCorrect: number;
  accuracy: number;
  hcMatches: number;
  hcCorrect: number;
  hcAccuracy: number;
  avgBrier: number;
}

export function getSeasonTotals(season: number): SeasonTotals {
  // Primary source: data/grading_history.json. EPL Oracle's DB lives in
  // git and isn't committed back from CI, so predictions.correct stays
  // NULL on every row and the DB query below always returned 0. The
  // python/recap.py script writes grading_history.json after each
  // gameweek; the morning Discord embed reads from it instead.
  const histTotals = readSeasonTotalsFromHistory(season);
  if (histTotals.totalMatches > 0) return histTotals;

  // Legacy DB path — kept for backward compatibility and in case some
  // environment populates predictions.correct directly.
  const rows = queryAll<{
    correct: number | null;
    calibrated_home_prob: number;
    calibrated_draw_prob: number;
    calibrated_away_prob: number;
  }>(
    `SELECT correct, calibrated_home_prob, calibrated_draw_prob, calibrated_away_prob
     FROM predictions WHERE season = ? AND correct IS NOT NULL`,
    [season],
  );

  const total = rows.length;
  const correct = rows.filter(r => r.correct === 1).length;
  const maxProb = (r: typeof rows[0]) => Math.max(r.calibrated_home_prob, r.calibrated_draw_prob, r.calibrated_away_prob);
  const hcRows = rows.filter(r => maxProb(r) >= 0.60);
  const hcCorrect = hcRows.filter(r => r.correct === 1).length;

  const brierRows = queryAll<{ avg_brier: number }>(
    `SELECT AVG(brier_score) as avg_brier FROM accuracy_log WHERE season = ?`,
    [season],
  );
  const avgBrier = brierRows[0]?.avg_brier ?? 0;

  return {
    season,
    totalMatches: total,
    totalCorrect: correct,
    accuracy: total > 0 ? correct / total : 0,
    hcMatches: hcRows.length,
    hcCorrect,
    hcAccuracy: hcRows.length > 0 ? hcCorrect / hcRows.length : 0,
    avgBrier,
  };
}

interface GradedPick {
  date: string;
  gameId: string;
  modelProb?: number;
  correct?: boolean;
  pickedTeam?: string;
}

function readSeasonTotalsFromHistory(season: number): SeasonTotals {
  const empty: SeasonTotals = {
    season, totalMatches: 0, totalCorrect: 0, accuracy: 0,
    hcMatches: 0, hcCorrect: 0, hcAccuracy: 0, avgBrier: 0,
  };
  try {
    // Lazy-load to avoid a hard fs dependency at module init.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { readFileSync, existsSync } = require('fs');
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const path = require('path');
    const file = path.resolve(process.cwd(), 'data/grading_history.json');
    if (!existsSync(file)) return empty;
    const parsed = JSON.parse(readFileSync(file, 'utf-8')) as { graded?: GradedPick[] };
    const graded = parsed.graded ?? [];
    // Filter to picks whose gameId's date matches the season window. EPL
    // seasons run August–May, so the simplest filter is: a graded entry
    // belongs to "season=N" if its date >= Aug 1 of year N or < Jul 1 of
    // year N+1. For the current 2025-26 season (season=2025) that's
    // 2025-08-01 .. 2026-07-31.
    const start = new Date(Date.UTC(season, 7, 1));  // Aug 1, year=season
    const end = new Date(Date.UTC(season + 1, 6, 31));  // Jul 31, year=season+1
    const inSeason = graded.filter(g => {
      const d = new Date(g.date + 'T00:00:00Z');
      return d >= start && d <= end;
    });
    if (inSeason.length === 0) return empty;
    const correct = inSeason.filter(g => g.correct).length;
    const hcPicks = inSeason.filter(g => (g.modelProb ?? 0) >= 0.60);
    const hcCorrect = hcPicks.filter(g => g.correct).length;
    return {
      season,
      totalMatches: inSeason.length,
      totalCorrect: correct,
      accuracy: correct / inSeason.length,
      hcMatches: hcPicks.length,
      hcCorrect,
      hcAccuracy: hcPicks.length > 0 ? hcCorrect / hcPicks.length : 0,
      avgBrier: 0,  // not tracked in the JSON; legacy DB path provided it
    };
  } catch {
    return empty;
  }
}

export interface ConfidenceBucket {
  label: string;     // e.g. "70-80%"
  total: number;
  correct: number;
  accuracy: number;  // 0..1
}

// Per-confidence-bucket calibration for the morning Discord embed. Bins
// graded picks from data/grading_history.json by their PICK-SIDE
// probability (modelProb is already pick-side: max(home, draw, away) —
// do NOT take max again, that would double-bucket). Buckets are
// half-open [lo, hi). Only buckets with at least one graded pick are
// returned so the embed stays compact early-season.
export function getConfidenceBuckets(season: number): ConfidenceBucket[] {
  const buckets: Array<{ lo: number; hi: number; label: string; total: number; correct: number }> = [
    { lo: 0.50, hi: 0.60, label: '50-60%', total: 0, correct: 0 },
    { lo: 0.60, hi: 0.70, label: '60-70%', total: 0, correct: 0 },
    { lo: 0.70, hi: 0.80, label: '70-80%', total: 0, correct: 0 },
    { lo: 0.80, hi: 0.90, label: '80-90%', total: 0, correct: 0 },
    { lo: 0.90, hi: 1.01, label: '90%+',   total: 0, correct: 0 },
  ];
  try {
    // Lazy-load to match readSeasonTotalsFromHistory's pattern.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { readFileSync, existsSync } = require('fs');
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const path = require('path');
    const file = path.resolve(process.cwd(), 'data/grading_history.json');
    if (!existsSync(file)) return [];
    const parsed = JSON.parse(readFileSync(file, 'utf-8')) as { graded?: GradedPick[] };
    const graded = parsed.graded ?? [];
    // Same season-window filter as readSeasonTotalsFromHistory: EPL
    // seasons run August–May, so season=N maps to 2025-08-01..2026-07-31.
    const start = new Date(Date.UTC(season, 7, 1));
    const end = new Date(Date.UTC(season + 1, 6, 31));
    const inSeason = graded.filter(g => {
      const d = new Date(g.date + 'T00:00:00Z');
      return d >= start && d <= end;
    });
    for (const g of inSeason) {
      const p = g.modelProb;
      if (p === undefined || p === null) continue;
      for (const b of buckets) {
        if (p >= b.lo && p < b.hi) {
          b.total += 1;
          if (g.correct) b.correct += 1;
          break;
        }
      }
    }
  } catch {
    return [];
  }
  return buckets
    .filter(b => b.total > 0)
    .map(b => ({
      label: b.label,
      total: b.total,
      correct: b.correct,
      accuracy: b.correct / b.total,
    }));
}

export function closeDb(): void {
  if (_db) {
    persistDb();
    _db.close();
    _db = null;
  }
}
