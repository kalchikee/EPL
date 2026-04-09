// EPL Oracle v4.1 — Odds API Client
// Uses The Odds API (free tier: 500 req/month) when ODDS_API_KEY is set.
// Soccer endpoint: soccer_epl (1X2 + totals markets)
// Falls back gracefully to ESPN embedded odds.

import fetch from 'node-fetch';
import { logger } from '../logger.js';

const ODDS_API_BASE = 'https://api.the-odds-api.com/v4/sports/soccer_epl/odds';

interface OddsAPIBookmaker {
  key: string;
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
  total: number;          // expected total goals (over/under line)
}

let _oddsCache: Map<string, MatchOdds> | null = null;

// Convert American moneyline to raw probability
function mlToProb(ml: number): number {
  if (ml > 0) return 100 / (ml + 100);
  return Math.abs(ml) / (Math.abs(ml) + 100);
}

// Convert decimal odds to American moneyline
function decimalToML(dec: number): number {
  if (dec >= 2.0) return Math.round((dec - 1) * 100);
  return Math.round(-100 / (dec - 1));
}

// Remove vig from three-way market
function removeVig3Way(rH: number, rD: number, rA: number): { home: number; draw: number; away: number } {
  const total = rH + rD + rA;
  return { home: rH / total, draw: rD / total, away: rA / total };
}

export async function loadOddsApiLines(): Promise<void> {
  const key = process.env.ODDS_API_KEY;
  if (!key) {
    logger.info('ODDS_API_KEY not set — skipping Odds API');
    return;
  }

  const url = `${ODDS_API_BASE}?apiKey=${key}&regions=uk,eu&markets=h2h,totals&oddsFormat=american&bookmakers=bet365,williamhill,paddypower`;

  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!resp.ok) {
      logger.warn({ status: resp.status }, 'Odds API request failed');
      return;
    }
    const games = (await resp.json()) as OddsAPIGame[];
    _oddsCache = new Map();

    for (const game of games) {
      const bk = game.bookmakers?.find(b => b.key === 'bet365')
        ?? game.bookmakers?.find(b => b.key === 'williamhill')
        ?? game.bookmakers?.[0];
      if (!bk) continue;

      const h2h = bk.markets.find(m => m.key === 'h2h');
      const totals = bk.markets.find(m => m.key === 'totals');

      if (!h2h) continue;

      // 1X2: home / draw / away
      const homeOutcome = h2h.outcomes.find(o => o.name === game.home_team);
      const drawOutcome = h2h.outcomes.find(o => o.name === 'Draw');
      const awayOutcome = h2h.outcomes.find(o => o.name === game.away_team);

      if (!homeOutcome || !awayOutcome) continue;

      const rawHome = mlToProb(homeOutcome.price);
      const rawDraw = drawOutcome ? mlToProb(drawOutcome.price) : 0.28;
      const rawAway = mlToProb(awayOutcome.price);
      const { home, draw, away } = removeVig3Way(rawHome, rawDraw, rawAway);

      const totalOver = totals?.outcomes.find(o => o.name === 'Over');
      const matchKey = `${game.away_team}@${game.home_team}`;

      _oddsCache.set(matchKey, {
        homeML: homeOutcome.price,
        drawML: drawOutcome?.price ?? -120,
        awayML: awayOutcome.price,
        homeImpliedProb: home,
        drawImpliedProb: draw,
        awayImpliedProb: away,
        total: totalOver?.point ?? 2.5,
      });
    }

    logger.info({ matches: _oddsCache.size }, 'Odds API lines loaded');
  } catch (err) {
    logger.warn({ err }, 'Failed to load Odds API lines');
  }
}

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
  // Try Odds API cache first (full team name match — approximate)
  if (_oddsCache) {
    for (const [key, odds] of _oddsCache.entries()) {
      if (key.toLowerCase().includes(homeAbbr.toLowerCase()) ||
          key.toLowerCase().includes(awayAbbr.toLowerCase())) {
        return odds;
      }
    }
  }

  // Fall back to ESPN embedded moneylines
  if (homeML && drawML && awayML) {
    const rawH = mlToProb(homeML);
    const rawD = mlToProb(drawML);
    const rawA = mlToProb(awayML);
    const { home, draw, away } = removeVig3Way(rawH, rawD, rawA);
    return {
      homeML, drawML, awayML,
      homeImpliedProb: home,
      drawImpliedProb: draw,
      awayImpliedProb: away,
      total: 2.5,
    };
  }

  // Fall back to ESPN decimal odds
  if (homeDecimal && drawDecimal && awayDecimal) {
    const hML = decimalToML(homeDecimal);
    const dML = decimalToML(drawDecimal);
    const aML = decimalToML(awayDecimal);
    const rawH = mlToProb(hML);
    const rawD = mlToProb(dML);
    const rawA = mlToProb(aML);
    const { home, draw, away } = removeVig3Way(rawH, rawD, rawA);
    return {
      homeML: hML, drawML: dML, awayML: aML,
      homeImpliedProb: home,
      drawImpliedProb: draw,
      awayImpliedProb: away,
      total: 2.5,
    };
  }

  return null;
}

export function hasAnyOdds(): boolean {
  return _oddsCache !== null && _oddsCache.size > 0;
}
