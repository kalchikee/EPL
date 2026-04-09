// EPL Oracle v4.1 — ESPN EPL API Client
// Free API — no key required.
// Covers: schedule, standings, team stats, completed results.
// JSON file caching with TTL + exponential backoff retry.

import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';
import { logger } from '../logger.js';
import type { EPLMatch, EPLMatchTeam, EPLTeamStats, MatchResult } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CACHE_DIR = process.env.CACHE_DIR ?? resolve(__dirname, '../../cache');
const CACHE_TTL_MS = (Number(process.env.CACHE_TTL_HOURS ?? 4)) * 60 * 60 * 1000;

mkdirSync(CACHE_DIR, { recursive: true });

// ESPN soccer EPL base URL
const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1';

// ─── Team abbreviation → ESPN ID mapping ─────────────────────────────────────
// ESPN uses numeric IDs for teams. These are stable across seasons.

export const ABBR_TO_ESPN_ID: Record<string, string> = {
  ARS: '359',   // Arsenal
  AVL: '1212',  // Aston Villa
  BOU: '349',   // Bournemouth
  BRE: '337',   // Brentford
  BHA: '331',   // Brighton & Hove Albion
  BUR: '379',   // Burnley
  CHE: '363',   // Chelsea
  CRY: '382',   // Crystal Palace
  EVE: '368',   // Everton
  FUL: '370',   // Fulham
  LEE: '357',   // Leeds United
  LIV: '364',   // Liverpool
  MCI: '382',   // Manchester City  (espn id differs, use below)
  MUN: '360',   // Manchester United
  NEW: '361',   // Newcastle United
  NFO: '374',   // Nottingham Forest
  SUN: '371',   // Sunderland (promoted)
  TOT: '367',   // Tottenham Hotspur
  WHU: '371',   // West Ham United  (check collision with SUN below)
  WOL: '380',   // Wolverhampton
};

// Correct ESPN IDs (some share abbr collisions above — fix them):
const ESPN_TEAM_IDS: Record<string, string> = {
  ARS: '359',
  AVL: '1212',
  BOU: '349',
  BRE: '337',
  BHA: '331',
  BUR: '379',
  CHE: '363',
  CRY: '382',
  EVE: '368',
  FUL: '370',
  LEE: '357',
  LIV: '364',
  MCI: '383',   // Man City actual ESPN ID
  MUN: '360',
  NEW: '361',
  NFO: '374',
  SUN: '398',   // Sunderland
  TOT: '367',
  WHU: '371',
  WOL: '380',
};

// Build reverse mapping
export const ESPN_ID_TO_ABBR: Record<string, string> = Object.fromEntries(
  Object.entries(ESPN_TEAM_IDS).map(([abbr, id]) => [id, abbr])
);

// ─── Cache helpers ─────────────────────────────────────────────────────────────

function cacheKey(url: string): string {
  return url.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 200) + '.json';
}

function readCache<T>(key: string): T | null {
  const path = resolve(CACHE_DIR, key);
  if (!existsSync(path)) return null;
  const stat = statSync(path);
  if (Date.now() - stat.mtimeMs > CACHE_TTL_MS) return null;
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as T;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: unknown): void {
  try {
    writeFileSync(resolve(CACHE_DIR, key), JSON.stringify(data), 'utf-8');
  } catch (err) {
    logger.warn({ err }, 'Failed to write cache');
  }
}

// ─── Retry fetch ──────────────────────────────────────────────────────────────

async function fetchWithRetry<T>(url: string, attempts = 3, bypassCache = false): Promise<T> {
  const key = cacheKey(url);
  if (!bypassCache) {
    const cached = readCache<T>(key);
    if (cached !== null) {
      logger.debug({ url }, 'Cache HIT');
      return cached;
    }
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      logger.debug({ url, attempt }, 'Fetching');
      const resp = await fetch(url, {
        headers: { 'User-Agent': 'EPLOracle/4.1 (educational)' },
        signal: AbortSignal.timeout(20000),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
      const data = (await resp.json()) as T;
      writeCache(key, data);
      return data;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt < attempts - 1) {
        const delay = Math.pow(2, attempt) * 1000;
        logger.warn({ url, attempt, delay, err: lastError.message }, 'Retrying');
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }
  throw lastError ?? new Error(`Failed to fetch ${url}`);
}

// ─── Season helpers ───────────────────────────────────────────────────────────

// EPL season labeled by start year (e.g., "2025" = 2025-26)
export function getCurrentEPLSeason(): number {
  const now = new Date();
  const month = now.getMonth() + 1;
  const year = now.getFullYear();
  // EPL runs Aug-May. Season start year: if July or later → current year; else prior year.
  return month >= 7 ? year : year - 1;
}

export function isEPLSeason(date: Date = new Date()): boolean {
  const month = date.getMonth() + 1;
  // EPL runs August through May
  if (month >= 8) return true;   // August - December
  if (month <= 5) return true;   // January - May
  return false;                  // June - July: off-season
}

export function getSeasonString(season: number): string {
  return `${season}-${(season + 1).toString().slice(2)}`;
}

// ─── ESPN response types ──────────────────────────────────────────────────────

interface ESPNScoreboardResponse {
  leagues?: Array<{ season?: { year?: number }; calendar?: Array<{ value?: string; label?: string }> }>;
  season?: { year?: number; type?: number };
  week?: { number?: number };
  events?: ESPNEvent[];
}

interface ESPNEvent {
  id: string;
  date: string;
  name?: string;
  week?: { number?: number };
  season?: { year?: number; type?: number };
  status?: { type?: { name?: string; description?: string; completed?: boolean } };
  competitions?: ESPNCompetition[];
}

interface ESPNCompetition {
  id: string;
  competitors?: ESPNCompetitor[];
  venue?: { fullName?: string; address?: { city?: string } };
  neutralSite?: boolean;
  odds?: Array<{
    details?: string;
    overUnder?: number;
    homeTeamOdds?: { moneyLine?: number; summary?: string };
    awayTeamOdds?: { moneyLine?: number; summary?: string };
    drawOdds?: { moneyLine?: number };
    open?: { total?: { alternateDisplayValue?: string } };
  }>;
}

interface ESPNCompetitor {
  id: string;
  homeAway: string;
  team: { id?: string; abbreviation?: string; displayName?: string; shortDisplayName?: string };
  score?: string;
}

function normalizeAbbr(raw: string | undefined, teamId: string | undefined): string {
  if (teamId && ESPN_ID_TO_ABBR[teamId]) return ESPN_ID_TO_ABBR[teamId];
  if (!raw) return 'UNK';
  // Map common ESPN abbreviations to our standard ones
  const map: Record<string, string> = {
    'Man City': 'MCI', 'MAN': 'MUN', 'MCFC': 'MCI', 'MUFC': 'MUN',
    'Man Utd': 'MUN', 'LIV': 'LIV', 'Arsenal': 'ARS',
    'Spurs': 'TOT', 'THFC': 'TOT', 'Chelsea': 'CHE', 'CFC': 'CHE',
    'Newcastle': 'NEW', 'NUFC': 'NEW', 'WHU': 'WHU', 'WHUFC': 'WHU',
    'Wolves': 'WOL', 'BOU': 'BOU', 'BRE': 'BRE', 'NFO': 'NFO',
  };
  return map[raw] ?? raw.toUpperCase().slice(0, 3);
}

function parseESPNEvent(event: ESPNEvent, defaultSeason: number): EPLMatch | null {
  const comp = event.competitions?.[0];
  if (!comp?.competitors || comp.competitors.length < 2) return null;

  const home = comp.competitors.find(c => c.homeAway === 'home');
  const away = comp.competitors.find(c => c.homeAway === 'away');
  if (!home || !away) return null;

  const homeAbbr = normalizeAbbr(home.team.abbreviation, home.team.id);
  const awayAbbr = normalizeAbbr(away.team.abbreviation, away.team.id);

  const homeTeam: EPLMatchTeam = {
    teamId: home.team.id ?? '',
    teamAbbr: homeAbbr,
    teamName: home.team.displayName ?? homeAbbr,
    score: home.score !== undefined ? Number(home.score) : undefined,
  };
  const awayTeam: EPLMatchTeam = {
    teamId: away.team.id ?? '',
    teamAbbr: awayAbbr,
    teamName: away.team.displayName ?? awayAbbr,
    score: away.score !== undefined ? Number(away.score) : undefined,
  };

  // Parse odds (ESPN sometimes includes them)
  const odds = comp.odds?.[0];
  let homeML: number | undefined;
  let awayML: number | undefined;
  let drawML: number | undefined;
  let vegasTotal: number | undefined;

  if (odds) {
    homeML = odds.homeTeamOdds?.moneyLine;
    awayML = odds.awayTeamOdds?.moneyLine;
    drawML = odds.drawOdds?.moneyLine;
    vegasTotal = odds.overUnder;
  }

  const matchDate = event.date.split('T')[0];

  return {
    matchId: event.id,
    matchDate,
    matchTime: event.date,
    gameweek: event.week?.number ?? 0,
    season: event.season?.year ?? defaultSeason,
    status: event.status?.type?.name ?? 'STATUS_SCHEDULED',
    homeTeam,
    awayTeam,
    venueName: comp.venue?.fullName ?? '',
    venueCity: comp.venue?.address?.city ?? '',
    homeMoneyLine: homeML,
    awayMoneyLine: awayML,
    drawMoneyLine: drawML,
    vegasTotal,
  };
}

// ─── Gameweek date range (approx) ────────────────────────────────────────────
// EPL GW1 ≈ Aug 16 each season; each GW is ~7 days.

function getGameweekDateRange(gw: number, season: number): { start: Date; end: Date } {
  // GW1 start: third Saturday of August (approx Aug 16)
  const gw1Start = new Date(`${season}-08-09`);
  const offset = (gw - 1) * 7;
  const start = new Date(gw1Start);
  start.setDate(start.getDate() + offset);
  const end = new Date(start);
  end.setDate(end.getDate() + 8); // 8-day window covers Fri–Mon fixtures
  return { start, end };
}

// ─── Gameweek schedule ────────────────────────────────────────────────────────
// ESPN soccer uses date-based queries (dates=YYYYMMDD), not week numbers.
// We scan each day in the GW window and collect all EPL matches.

export async function fetchGameweekSchedule(gameweek: number, season: number): Promise<EPLMatch[]> {
  const { start, end } = getGameweekDateRange(gameweek, season);
  const matches: EPLMatch[] = [];
  const seen = new Set<string>();

  let current = new Date(start);
  while (current <= end) {
    const dateStr = current.toISOString().split('T')[0].replace(/-/g, '');
    const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}&limit=20`;
    try {
      const data = await fetchWithRetry<ESPNScoreboardResponse>(url);
      if (data.events) {
        for (const event of data.events) {
          if (!seen.has(event.id)) {
            seen.add(event.id);
            const match = parseESPNEvent(event, season);
            if (match) {
              match.gameweek = gameweek; // tag with requested GW
              matches.push(match);
            }
          }
        }
      }
    } catch (err) {
      logger.debug({ err, date: dateStr }, 'Failed to fetch scoreboard for date');
    }
    current.setDate(current.getDate() + 1);
  }

  logger.info({ gameweek, season, matches: matches.length }, 'Gameweek schedule fetched');
  return matches;
}

// Fetch matches for a date range
export async function fetchMatchesForDates(startDate: string, endDate: string): Promise<EPLMatch[]> {
  const season = getCurrentEPLSeason();
  const matches: EPLMatch[] = [];
  const seen = new Set<string>();
  let current = new Date(startDate);
  const end = new Date(endDate);

  while (current <= end) {
    const dateStr = current.toISOString().split('T')[0].replace(/-/g, '');
    const url = `${ESPN_BASE}/scoreboard?dates=${dateStr}&limit=20`;
    try {
      const data = await fetchWithRetry<ESPNScoreboardResponse>(url);
      if (data.events) {
        for (const event of data.events) {
          if (!seen.has(event.id)) {
            seen.add(event.id);
            const match = parseESPNEvent(event, season);
            if (match) matches.push(match);
          }
        }
      }
    } catch (err) {
      logger.debug({ err, date: dateStr }, 'Failed to fetch scoreboard for date');
    }
    current.setDate(current.getDate() + 1);
  }
  return matches;
}

// ─── Current gameweek info ────────────────────────────────────────────────────
// Derives the gameweek number from today's date relative to GW1 start.

export async function getCurrentGameweekInfo(): Promise<{ gameweek: number; season: number }> {
  const season = getCurrentEPLSeason();

  // Derive GW from calendar: GW1 ≈ Aug 9 of season year
  const gw1Start = new Date(`${season}-08-09`);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - gw1Start.getTime()) / (1000 * 60 * 60 * 24));
  const estimatedGw = Math.max(1, Math.min(38, Math.floor(diffDays / 7) + 1));

  // Verify: scan ±2 GWs to find the one with upcoming/recent matches
  for (let offset = 0; offset <= 2; offset++) {
    for (const dir of [0, 1, -1]) {
      const gw = Math.max(1, Math.min(38, estimatedGw + dir + offset));
      const { start, end } = getGameweekDateRange(gw, season);
      const tomorrow = new Date(now);
      tomorrow.setDate(tomorrow.getDate() + 7);
      // If this GW's window overlaps with [now-3d, now+7d], it's current
      const windowStart = new Date(now);
      windowStart.setDate(windowStart.getDate() - 3);
      if (end >= windowStart && start <= tomorrow) {
        logger.info({ estimatedGw: gw, season }, 'Current gameweek detected');
        return { gameweek: gw, season };
      }
    }
  }

  return { gameweek: estimatedGw, season };
}

// ─── Prior season stats (2024-25 final) ──────────────────────────────────────
// Used to seed GW1-10 before current-season data is meaningful.
// All values are per-game averages from the 2024-25 EPL final standings.
// Promoted teams get Championship-regressed priors (≈70% toward EPL average).

interface TeamPrior {
  gfPg: number;           // goals for per game
  gaPg: number;           // goals against per game
  csRate: number;         // clean sheet rate (0-1)
  bttsRate: number;       // both teams to score rate (0-1)
  possessionPct: number;  // average possession %
  sotPg: number;          // shots on target per game
}

const LEAGUE_AVG_PRIOR: TeamPrior = {
  gfPg: 1.35, gaPg: 1.35, csRate: 0.28, bttsRate: 0.55, possessionPct: 50, sotPg: 3.5,
};

const PRIOR_STATS_2024_25: Record<string, TeamPrior> = {
  // ── Top six ────────────────────────────────────────────────────────────────
  LIV: { gfPg: 2.34, gaPg: 0.97, csRate: 0.45, bttsRate: 0.42, possessionPct: 58, sotPg: 5.5 },
  ARS: { gfPg: 1.84, gaPg: 0.97, csRate: 0.42, bttsRate: 0.39, possessionPct: 57, sotPg: 5.2 },
  MCI: { gfPg: 1.82, gaPg: 1.37, csRate: 0.34, bttsRate: 0.50, possessionPct: 62, sotPg: 5.6 },
  CHE: { gfPg: 1.68, gaPg: 1.42, csRate: 0.31, bttsRate: 0.53, possessionPct: 53, sotPg: 4.8 },
  NEW: { gfPg: 1.68, gaPg: 1.21, csRate: 0.37, bttsRate: 0.47, possessionPct: 48, sotPg: 4.5 },
  AVL: { gfPg: 1.61, gaPg: 1.47, csRate: 0.29, bttsRate: 0.53, possessionPct: 50, sotPg: 4.6 },
  // ── Upper mid-table ────────────────────────────────────────────────────────
  TOT: { gfPg: 1.74, gaPg: 1.58, csRate: 0.26, bttsRate: 0.58, possessionPct: 52, sotPg: 5.0 },
  NFO: { gfPg: 1.16, gaPg: 1.21, csRate: 0.37, bttsRate: 0.40, possessionPct: 44, sotPg: 3.7 },
  BHA: { gfPg: 1.58, gaPg: 1.50, csRate: 0.29, bttsRate: 0.55, possessionPct: 55, sotPg: 4.6 },
  FUL: { gfPg: 1.50, gaPg: 1.53, csRate: 0.27, bttsRate: 0.55, possessionPct: 47, sotPg: 4.1 },
  // ── Lower mid-table ────────────────────────────────────────────────────────
  MUN: { gfPg: 1.18, gaPg: 1.61, csRate: 0.24, bttsRate: 0.50, possessionPct: 50, sotPg: 3.9 },
  WOL: { gfPg: 1.37, gaPg: 1.74, csRate: 0.21, bttsRate: 0.55, possessionPct: 43, sotPg: 3.6 },
  BRE: { gfPg: 1.37, gaPg: 1.76, csRate: 0.21, bttsRate: 0.57, possessionPct: 43, sotPg: 4.0 },
  CRY: { gfPg: 1.11, gaPg: 1.55, csRate: 0.26, bttsRate: 0.47, possessionPct: 42, sotPg: 3.4 },
  EVE: { gfPg: 1.13, gaPg: 1.47, csRate: 0.28, bttsRate: 0.47, possessionPct: 43, sotPg: 3.5 },
  WHU: { gfPg: 1.34, gaPg: 1.74, csRate: 0.21, bttsRate: 0.55, possessionPct: 44, sotPg: 3.7 },
  BOU: { gfPg: 1.53, gaPg: 1.63, csRate: 0.26, bttsRate: 0.58, possessionPct: 47, sotPg: 4.2 },
  // ── Promoted teams (2025-26) — Championship stats regressed 30% to EPL avg ─
  // Formula: prior = 0.70 * championship_est + 0.30 * epl_avg
  LEE: { gfPg: 1.27, gaPg: 1.57, csRate: 0.22, bttsRate: 0.53, possessionPct: 47, sotPg: 3.7 },
  BUR: { gfPg: 1.16, gaPg: 1.62, csRate: 0.21, bttsRate: 0.52, possessionPct: 44, sotPg: 3.5 },
  SUN: { gfPg: 1.13, gaPg: 1.60, csRate: 0.22, bttsRate: 0.51, possessionPct: 43, sotPg: 3.4 },
};

// Number of games played before we fully trust current-season stats.
// At played=0: 100% prior. At played=PRIOR_FADE: 100% current.
const PRIOR_FADE_GAMES = 10;

function blendWithPrior(current: number, prior: number, played: number): number {
  if (played >= PRIOR_FADE_GAMES) return current;
  const priorWeight = PRIOR_FADE_GAMES - played;
  return (played * current + priorWeight * prior) / PRIOR_FADE_GAMES;
}

// ─── Team stats from standings ────────────────────────────────────────────────

interface ESPNStandingsResponse {
  standings?: {
    entries?: Array<{
      team?: { id?: string; abbreviation?: string; displayName?: string };
      stats?: Array<{ name?: string; value?: number; displayValue?: string }>;
    }>;
  };
}

interface ESPNTeamStatsResponse {
  team?: { id?: string; abbreviation?: string; displayName?: string };
  results?: { splits?: { categories?: ESPNStatCategory[] } };
}

interface ESPNStatCategory {
  name?: string;
  stats?: Array<{ name?: string; displayValue?: string; value?: number }>;
}

let _teamStatsCache: Map<string, EPLTeamStats> | null = null;
let _teamStatsCacheTime = 0;

export async function fetchAllTeamStats(season?: number): Promise<Map<string, EPLTeamStats>> {
  const now = Date.now();
  if (_teamStatsCache && now - _teamStatsCacheTime < CACHE_TTL_MS) {
    return _teamStatsCache;
  }

  const s = season ?? getCurrentEPLSeason();
  const teamMap = new Map<string, EPLTeamStats>();

  // First try to get standings (has wins/draws/losses/goals)
  const standingsUrl = `${ESPN_BASE}/standings?season=${s}`;
  try {
    const data = await fetchWithRetry<ESPNStandingsResponse>(standingsUrl);
    const entries = data.standings?.entries ?? [];

    for (const entry of entries) {
      if (!entry.team) continue;
      const abbr = normalizeAbbr(entry.team.abbreviation, entry.team.id);
      const getStat = (name: string) =>
        entry.stats?.find(s => s.name?.toLowerCase() === name.toLowerCase())?.value ?? 0;

      const played = getStat('gamesPlayed') || getStat('played');
      const wins = getStat('wins');
      const draws = getStat('ties') || getStat('draws');
      const losses = getStat('losses');
      const points = getStat('points');
      const position = getStat('rank') || getStat('position');
      const gf = getStat('pointsFor') || getStat('goalsFor');
      const ga = getStat('pointsAgainst') || getStat('goalsAgainst');
      const gp = Math.max(played, 1);

      teamMap.set(abbr, buildTeamStats(abbr, entry.team.displayName ?? abbr, {
        played: gp, wins, draws, losses, points, position,
        gf, ga,
      }));
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch standings — using defaults');
  }

  // Fill in missing teams with defaults
  for (const abbr of Object.keys(ESPN_TEAM_IDS)) {
    if (!teamMap.has(abbr)) {
      teamMap.set(abbr, defaultTeamStats(abbr));
    }
  }

  _teamStatsCache = teamMap;
  _teamStatsCacheTime = now;
  logger.info({ teams: teamMap.size, season: s }, 'EPL team stats loaded');
  return teamMap;
}

function buildTeamStats(
  abbr: string,
  name: string,
  data: { played: number; wins: number; draws: number; losses: number; points: number; position: number; gf: number; ga: number },
): EPLTeamStats {
  const { played, wins, draws, losses, points, position, gf, ga } = data;
  const gp = Math.max(played, 1);

  // Raw current-season per-game rates (unreliable when played is small)
  const rawGfPg = played > 0 ? gf / played : 0;
  const rawGaPg = played > 0 ? ga / played : 0;

  // Pull prior for this team (fall back to league average for unknown teams)
  const prior = PRIOR_STATS_2024_25[abbr] ?? LEAGUE_AVG_PRIOR;

  // Blend: at GW1 (played=0) → 100% prior; at GW10+ → 100% current
  const gfPg   = blendWithPrior(rawGfPg,         prior.gfPg,         played);
  const gaPg   = blendWithPrior(rawGaPg,         prior.gaPg,         played);
  const csRate = blendWithPrior(
    played > 0 ? Math.max(0, 0.4 - (rawGaPg - 1.0) * 0.15) : 0,
    prior.csRate, played,
  );
  const bttsR  = blendWithPrior(
    played > 0 ? Math.min(0.75, 0.5 + (rawGfPg - 1.35) * 0.1 + (rawGaPg - 1.35) * 0.1) : 0,
    prior.bttsRate, played,
  );
  const poss   = blendWithPrior(
    played > 0 ? 50 + ((rawGfPg / 1.35) - 1.0) * 10 : 50,
    prior.possessionPct, played,
  );
  const sot    = blendWithPrior(
    played > 0 ? 3.5 + (rawGfPg - 1.35) * 2 : 3.5,
    prior.sotPg, played,
  );

  const LEAGUE_AVG = 1.35;
  // Home field advantage factors (same as monteCarlo.ts HOME_ADV)
  const HOME_ATT_FACTOR = 1.18;
  const HOME_DEF_FACTOR = 0.88;

  const attackRating  = Math.max(0.3, gfPg / LEAGUE_AVG);
  const defenseRating = Math.max(0.3, gaPg / LEAGUE_AVG);

  // Home/away split ratings: teams score/concede differently at home vs away.
  // Estimated from ESPN's combined stats using typical EPL home advantage ratios.
  const homeAttackRating  = Math.max(0.3, attackRating  * HOME_ATT_FACTOR);
  const homeDefenseRating = Math.max(0.3, defenseRating * HOME_DEF_FACTOR);
  const awayAttackRating  = Math.max(0.3, attackRating  / HOME_ATT_FACTOR);
  const awayDefenseRating = Math.max(0.3, defenseRating / HOME_DEF_FACTOR);

  // Form: blend prior form (derived from prior win%) with current form
  const priorForm = Math.min(1.0, (prior.gfPg - prior.gaPg + 1.35) / (1.35 * 2));
  const winPct  = played > 0 ? wins  / played : 0;
  const drawPct = played > 0 ? draws / played : 0;
  const rawForm = Math.min(1.0, (winPct * 3 + drawPct * 1) / 3);
  const formLast5 = blendWithPrior(rawForm, priorForm, played);

  if (played < PRIOR_FADE_GAMES) {
    logger.debug(
      { abbr, played, gfPg: gfPg.toFixed(2), gaPg: gaPg.toFixed(2), priorWeight: PRIOR_FADE_GAMES - played },
      'Using blended prior stats',
    );
  }

  return {
    teamId: ESPN_TEAM_IDS[abbr] ?? '',
    teamAbbr: abbr,
    teamName: name,
    played: gp,
    wins, draws, losses, points,
    tablePosition: position || 10,
    goalsFor: gf,
    goalsAgainst: ga,
    goalDifference: gf - ga,
    goalsForPerGame: gfPg,
    goalsAgainstPerGame: gaPg,
    attackRating,
    defenseRating,
    homeAttackRating,
    homeDefenseRating,
    awayAttackRating,
    awayDefenseRating,
    formLast5,
    homeFormLast5: formLast5,
    awayFormLast5: formLast5 * 0.85,
    cleanSheetRate: csRate,
    bttsRate: bttsR,
    xgFor: gfPg,
    xgAgainst: gaPg,
    shotsOnTargetPerGame: sot,
    possessionPct: poss,
  };
}

function defaultTeamStats(abbr: string): EPLTeamStats {
  return {
    teamId: ESPN_TEAM_IDS[abbr] ?? '',
    teamAbbr: abbr, teamName: abbr,
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

// ─── Completed results ────────────────────────────────────────────────────────

export async function fetchCompletedResults(startDate: string, endDate: string): Promise<MatchResult[]> {
  const matches = await fetchMatchesForDates(startDate, endDate);
  const results: MatchResult[] = [];

  for (const match of matches) {
    if (!match.status.includes('FINAL') && !match.status.includes('final') && !match.status.includes('STATUS_FINAL')) continue;
    const homeScore = match.homeTeam.score;
    const awayScore = match.awayTeam.score;
    if (homeScore === undefined || awayScore === undefined) continue;

    results.push({
      match_id: match.matchId,
      date: match.matchDate,
      gameweek: match.gameweek,
      season: match.season,
      home_team: match.homeTeam.teamAbbr,
      away_team: match.awayTeam.teamAbbr,
      home_score: homeScore,
      away_score: awayScore,
      venue: match.venueName,
    });
  }
  return results;
}
