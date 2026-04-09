// EPL Oracle v4.2 — Odds API Client
// Uses The Odds API (free tier: 500 req/month) when ODDS_API_KEY is set.
// Averages across ALL available bookmakers (including Pinnacle) after vig removal.
// Falls back to ESPN embedded odds.

import fetch from 'node-fetch';
import { logger } from '../logger.js';

const ODDS_API_BASE = 'https://api.the-odds-api.com/v4/sports/soccer_epl/odds';

// ─── Team name → abbreviation (The Odds API uses full names) ─────────────────
// Source: The Odds API returns names like "Manchester City", "Liverpool", etc.

const ODDS_API_NAME_TO_ABBR: Record<string, string> = {
  'manchester city':          'MCI',
  'manchester united':        'MUN',
  'man city':                 'MCI',
  'man utd':                  'MUN',
  'man united':               'MUN',
  'liverpool':                'LIV',
  'arsenal':                  'ARS',
  'chelsea':                  'CHE',
  'tottenham hotspur':        'TOT',
  'tottenham':                'TOT',
  'spurs':                    'TOT',
  'newcastle united':         'NEW',
  'newcastle':                'NEW',
  'aston villa':              'AVL',
  'brighton & hove albion':   'BHA',
  'brighton and hove albion': 'BHA',
  'brighton':                 'BHA',
  'wolverhampton wanderers':  'WOL',
  'wolverhampton':            'WOL',
  'wolves':                   'WOL',
  'west ham united':          'WHU',
  'west ham':                 'WHU',
  'crystal palace':           'CRY',
  'everton':                  'EVE',
  'brentford':                'BRE',
  'fulham':                   'FUL',
  'nottingham forest':        'NFO',
  'bournemouth':              'BOU',
  'afc bournemouth':          'BOU',
  'leeds united':             'LEE',
  'leeds':                    'LEE',
  'burnley':                  'BUR',
  'luton town':               'LUT',
  'luton':                    'LUT',
  'sheffield united':         'SHU',
  'leicester city':           'LEI',
  'leicester':                'LEI',
  'southampton':              'SOU',
  'ipswich town':             'IPS',
  'ipswich':                  'IPS',
  'sunderland':               'SUN',
  'norwich city':             'NOR',
  'watford':                  'WAT',
};

function nameToAbbr(name: string): string | null {
  return ODDS_API_NAME_TO_ABBR[name.toLowerCase().trim()] ?? null;
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface OddsAPIBookmaker {
  key: string;
  title: string;
  markets: Array<{
    key: string;
    outcomes: Array<{ name: string; price: number; point?: number }>;
  }>;
}

interface OddsAPIGame {
  id: string;
  home_team: string;
  away_team: string;
  bookmakers?: OddsAPIBookmaker[];
}

export interface MatchOdds {
  homeML: number;
  drawML: number;
  awayML: number;
  homeImpliedProb: number;
  drawImpliedProb: number;
  awayImpliedProb: number;
  total: number;
  bookmakerCount: number;    // how many books contributed to this line
}

// Cache keyed by "HOME_abbr@AWAY_abbr"
let _oddsCache: Map<string, MatchOdds> | null = null;

// ─── Math helpers ─────────────────────────────────────────────────────────────

function mlToProb(ml: number): number {
  if (ml > 0) return 100 / (ml + 100);
  return Math.abs(ml) / (Math.abs(ml) + 100);
}

function decimalToML(dec: number): number {
  if (dec >= 2.0) return Math.round((dec - 1) * 100);
  return Math.round(-100 / (dec - 1));
}

function decimalToProb(dec: number): number {
  return dec > 1 ? 1 / dec : 0;
}

function removeVig3Way(rH: number, rD: number, rA: number): { home: number; draw: number; away: number } {
  const total = rH + rD + rA;
  if (total <= 0) return { home: 0.45, draw: 0.27, away: 0.28 };
  return { home: rH / total, draw: rD / total, away: rA / total };
}

// ─── Load odds from The Odds API ──────────────────────────────────────────────

export async function loadOddsApiLines(): Promise<void> {
  const key = process.env.ODDS_API_KEY;
  if (!key) {
    logger.info('ODDS_API_KEY not set — skipping Odds API');
    return;
  }

  // Request all UK/EU bookmakers including Pinnacle (sharpest closing line)
  const url = `${ODDS_API_BASE}?apiKey=${key}&regions=uk,eu,us&markets=h2h,totals&oddsFormat=decimal`;

  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!resp.ok) {
      const text = await resp.text();
      logger.warn({ status: resp.status, body: text.slice(0, 200) }, 'Odds API request failed');
      return;
    }

    const games = (await resp.json()) as OddsAPIGame[];
    _oddsCache = new Map();

    for (const game of games) {
      const homeAbbr = nameToAbbr(game.home_team);
      const awayAbbr = nameToAbbr(game.away_team);

      if (!homeAbbr || !awayAbbr) {
        logger.debug({ home: game.home_team, away: game.away_team }, 'Odds API: unknown team names');
        continue;
      }

      // Collect vig-removed probs from ALL available bookmakers, then average
      const probSets: Array<{ home: number; draw: number; away: number }> = [];
      let bestTotal = 2.5;
      let bestML = { home: 0, draw: 0, away: 0 };

      for (const bk of game.bookmakers ?? []) {
        const h2h = bk.markets.find(m => m.key === 'h2h');
        if (!h2h) continue;

        const homeOut = h2h.outcomes.find(o => o.name === game.home_team);
        const drawOut = h2h.outcomes.find(o => o.name === 'Draw');
        const awayOut = h2h.outcomes.find(o => o.name === game.away_team);

        if (!homeOut || !drawOut || !awayOut) continue;

        const rH = decimalToProb(homeOut.price);
        const rD = decimalToProb(drawOut.price);
        const rA = decimalToProb(awayOut.price);
        const probs = removeVig3Way(rH, rD, rA);
        probSets.push(probs);

        // Store ML from first bookmaker (for display)
        if (probSets.length === 1) {
          bestML = {
            home: decimalToML(homeOut.price),
            draw: decimalToML(drawOut.price),
            away: decimalToML(awayOut.price),
          };
        }

        // Get total line from any bookmaker
        const totals = bk.markets.find(m => m.key === 'totals');
        if (totals) {
          const over = totals.outcomes.find(o => o.name === 'Over');
          if (over?.point) bestTotal = over.point;
        }
      }

      if (probSets.length === 0) continue;

      // Average across all bookmakers (consensus removes individual book bias)
      const avgHome = probSets.reduce((s, p) => s + p.home, 0) / probSets.length;
      const avgDraw = probSets.reduce((s, p) => s + p.draw, 0) / probSets.length;
      const avgAway = probSets.reduce((s, p) => s + p.away, 0) / probSets.length;

      const matchKey = `${homeAbbr}@${awayAbbr}`;
      _oddsCache.set(matchKey, {
        homeML: bestML.home,
        drawML: bestML.draw,
        awayML: bestML.away,
        homeImpliedProb: avgHome,
        drawImpliedProb: avgDraw,
        awayImpliedProb: avgAway,
        total: bestTotal,
        bookmakerCount: probSets.length,
      });

      logger.debug(
        { home: homeAbbr, away: awayAbbr, books: probSets.length, homeProb: avgHome.toFixed(3) },
        'Odds API line loaded',
      );
    }

    logger.info({ matches: _oddsCache.size }, 'Odds API lines loaded');
  } catch (err) {
    logger.warn({ err }, 'Failed to load Odds API lines');
  }
}

// ─── Retrieve odds for a specific match ───────────────────────────────────────

export function getOddsForMatch(
  homeAbbr: string,
  awayAbbr: string,
  homeDecimal?: number,
  drawDecimal?: number,
  awayDecimal?: number,
  homeML?: number,
  drawML?: number,
  awayML?: number,
): MatchOdds | null {
  // Try Odds API cache first (keyed by abbr pair — exact match)
  if (_oddsCache) {
    const key = `${homeAbbr}@${awayAbbr}`;
    const cached = _oddsCache.get(key);
    if (cached) return cached;
  }

  // Fall back to ESPN embedded moneylines
  if (homeML && drawML && awayML) {
    const rawH = mlToProb(homeML);
    const rawD = mlToProb(drawML);
    const rawA = mlToProb(awayML);
    const { home, draw, away } = removeVig3Way(rawH, rawD, rawA);
    return { homeML, drawML, awayML, homeImpliedProb: home, drawImpliedProb: draw, awayImpliedProb: away, total: 2.5, bookmakerCount: 1 };
  }

  // Fall back to ESPN decimal odds
  if (homeDecimal && drawDecimal && awayDecimal && homeDecimal > 1) {
    const hML = decimalToML(homeDecimal);
    const dML = decimalToML(drawDecimal);
    const aML = decimalToML(awayDecimal);
    const rawH = decimalToProb(homeDecimal);
    const rawD = decimalToProb(drawDecimal);
    const rawA = decimalToProb(awayDecimal);
    const { home, draw, away } = removeVig3Way(rawH, rawD, rawA);
    return { homeML: hML, drawML: dML, awayML: aML, homeImpliedProb: home, drawImpliedProb: draw, awayImpliedProb: away, total: 2.5, bookmakerCount: 1 };
  }

  return null;
}

export function hasAnyOdds(): boolean {
  return _oddsCache !== null && _oddsCache.size > 0;
}
