// EPL Oracle v4.2 — Injury Client (API-Football)
// Optional: set API_FOOTBALL_KEY in .env for injury data.
// Free tier: 100 requests/day — easily covers our 3 runs/week.
// Used purely for Discord alert enhancement + post-model probability adjustment.
// NOT a training feature (no historical injury data available for training).
//
// Sign up free at: https://www.api-football.com/

import fetch from 'node-fetch';
import { logger } from '../logger.js';

const API_BASE = 'https://v3.football.api-sports.io';
const EPL_LEAGUE_ID = 39;

// API-Football team IDs for EPL
const APIFOOTBALL_TEAM_IDS: Record<string, number> = {
  ARS: 42,    // Arsenal
  AVL: 66,    // Aston Villa
  BOU: 35,    // Bournemouth
  BRE: 55,    // Brentford
  BHA: 51,    // Brighton
  BUR: 44,    // Burnley
  CHE: 49,    // Chelsea
  CRY: 52,    // Crystal Palace
  EVE: 45,    // Everton
  FUL: 36,    // Fulham
  LEE: 63,    // Leeds United
  LEI: 46,    // Leicester City
  LIV: 40,    // Liverpool
  MCI: 50,    // Manchester City
  MUN: 33,    // Manchester United
  NEW: 34,    // Newcastle United
  NFO: 65,    // Nottingham Forest
  SOU: 41,    // Southampton
  TOT: 47,    // Tottenham Hotspur
  WHU: 48,    // West Ham United
  WOL: 39,    // Wolverhampton
  IPS: 57,    // Ipswich Town
  SHU: 62,    // Sheffield United
  LUT: 1359,  // Luton Town
  SUN: 72,    // Sunderland
};

// Position impact weights for squad strength calculation
const POSITION_WEIGHTS: Record<string, number> = {
  'Attacker':   0.035,  // forwards: highest impact
  'Midfielder': 0.020,
  'Defender':   0.018,
  'Goalkeeper': 0.028,  // losing starting GK is significant
};

export interface InjuredPlayer {
  name: string;
  position: string;
  injuryType: 'Injured' | 'Suspended' | 'Questionable';
  reason: string;     // e.g. "Hamstring", "5th yellow card"
}

export interface TeamInjuries {
  abbr: string;
  players: InjuredPlayer[];
  squadStrength: number;  // 0.75–1.0 (1.0 = fully fit, lower = key players out)
  label: string;          // human-readable summary e.g. "Nunez (hamstring), Saka (ankle)"
}

// Cache for the current run
let _injuryCache: Map<string, TeamInjuries> | null = null;

// ─── Fetch injuries for all EPL teams playing this gameweek ──────────────────

export async function fetchInjuries(season: number): Promise<Map<string, TeamInjuries>> {
  const apiKey = process.env.API_FOOTBALL_KEY;
  if (!apiKey) {
    logger.debug('API_FOOTBALL_KEY not set — skipping injury data');
    return new Map();
  }

  if (_injuryCache) return _injuryCache;

  _injuryCache = new Map();

  // Fetch all current EPL injuries in one call (not per-team)
  const url = `${API_BASE}/injuries?league=${EPL_LEAGUE_ID}&season=${season}`;

  try {
    const resp = await fetch(url, {
      headers: {
        'x-rapidapi-host': 'v3.football.api-sports.io',
        'x-rapidapi-key': apiKey,
      },
      signal: AbortSignal.timeout(15000),
    });

    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'API-Football injury request failed');
      return _injuryCache;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const data = (await resp.json()) as any;
    const responses = data?.response ?? [];

    // Group by team
    const byTeam = new Map<string, InjuredPlayer[]>();
    const teamIdToAbbr = new Map<number, string>(
      Object.entries(APIFOOTBALL_TEAM_IDS).map(([abbr, id]) => [id, abbr])
    );

    for (const entry of responses) {
      const teamId = entry?.team?.id;
      const abbr = teamIdToAbbr.get(teamId);
      if (!abbr) continue;

      const playerName = entry?.player?.name ?? 'Unknown';
      const pos = entry?.player?.type ?? 'Midfielder';  // position is in 'type'
      const injuryType = (entry?.player?.reason ?? '').includes('Yellow') ? 'Suspended'
        : entry?.player?.type === 'Questionable' ? 'Questionable'
        : 'Injured';
      const reason = entry?.player?.reason ?? pos;

      if (!byTeam.has(abbr)) byTeam.set(abbr, []);
      byTeam.get(abbr)!.push({
        name: playerName,
        position: pos,
        injuryType,
        reason,
      });
    }

    // Build TeamInjuries with squad strength calculation
    for (const [abbr, players] of byTeam.entries()) {
      const teamInjuries = buildTeamInjuries(abbr, players);
      _injuryCache.set(abbr, teamInjuries);
    }

    logger.info({ teams: _injuryCache.size }, 'Injury data loaded from API-Football');
  } catch (err) {
    logger.warn({ err }, 'Failed to fetch injury data');
  }

  return _injuryCache;
}

function buildTeamInjuries(abbr: string, players: InjuredPlayer[]): TeamInjuries {
  // Cap to top 5 by severity (attackers and GKs first)
  const sorted = [...players].sort((a, b) => {
    const wa = POSITION_WEIGHTS[a.position] ?? 0.02;
    const wb = POSITION_WEIGHTS[b.position] ?? 0.02;
    return wb - wa;
  });
  const top5 = sorted.slice(0, 5);

  // Squad strength: 1.0 = fully fit, subtract per injury
  let deduction = 0;
  for (const p of top5) {
    deduction += POSITION_WEIGHTS[p.position] ?? 0.02;
    if (p.injuryType === 'Questionable') deduction *= 0.5; // half weight for maybes
  }
  const squadStrength = Math.max(0.75, 1.0 - deduction);

  const label = top5.length > 0
    ? top5.map(p => `${p.name.split(' ').pop()} (${p.reason.toLowerCase()})`).join(', ')
    : 'No major injuries';

  return { abbr, players: top5, squadStrength, label };
}

// ─── Get squad strength for a specific team ───────────────────────────────────

export function getTeamInjuries(abbr: string): TeamInjuries | null {
  return _injuryCache?.get(abbr) ?? null;
}

// ─── Compute injury-based probability adjustment ──────────────────────────────
// Adjusts calibrated probabilities based on relative squad fitness.
// This is applied AFTER the ML model as a final adjustment.

export function applyInjuryAdjustment(
  homeAbbr: string,
  awayAbbr: string,
  calibrated: { home: number; draw: number; away: number },
): { home: number; draw: number; away: number } {
  const homeInjuries = getTeamInjuries(homeAbbr);
  const awayInjuries = getTeamInjuries(awayAbbr);

  if (!homeInjuries && !awayInjuries) return calibrated;

  const homeStrength = homeInjuries?.squadStrength ?? 1.0;
  const awayStrength = awayInjuries?.squadStrength ?? 1.0;
  const diff = homeStrength - awayStrength;  // positive = home is fitter

  // Scale: 0.10 fitness advantage ≈ 2.0% probability shift
  const maxAdjustment = 0.04;  // cap at ±4%
  const adjustment = Math.max(-maxAdjustment, Math.min(maxAdjustment, diff * 0.20));

  // Shift win probabilities; draw absorbs some of the redistribution
  let { home, draw, away } = calibrated;
  home  = Math.max(0.02, Math.min(0.95, home  + adjustment));
  away  = Math.max(0.02, Math.min(0.95, away  - adjustment * 0.7));
  draw  = Math.max(0.02, Math.min(0.95, draw  - adjustment * 0.3));

  // Renormalize
  const total = home + draw + away;
  return { home: home / total, draw: draw / total, away: away / total };
}
