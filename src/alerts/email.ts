// EPL Oracle v4.1 — Email Alerts via Resend API
// Sends picks briefing + recap to email alongside Discord.

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getConfidenceTier, getDominantOutcome } from '../features/marketEdge.js';
import type { Prediction } from '../types.js';
import type { SeasonTotals } from '../db/database.js';
import type { MatchWithResult, RecapMetrics } from './results.js';

// ─── Config ───────────────────────────────────────────────────────────────────

const RESEND_API_URL = 'https://api.resend.com/emails';
const FROM_ADDRESS   = 'EPL Oracle <picks@kalchi.com>';
const TO_ADDRESS     = process.env.EMAIL_TO ?? 'kalchi.picks@gmail.com';

// ─── Resend sender ────────────────────────────────────────────────────────────

async function sendEmail(subject: string, html: string): Promise<boolean> {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    logger.warn('RESEND_API_KEY not set — skipping email alert');
    return false;
  }
  try {
    const resp = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ from: FROM_ADDRESS, to: [TO_ADDRESS], subject, html }),
      signal: AbortSignal.timeout(15000),
    });
    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Resend email error');
      return false;
    }
    logger.info({ to: TO_ADDRESS, subject }, 'EPL email sent');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send EPL email');
    return false;
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function getSeasonLabel(season: number): string {
  return `${season}-${(season + 1).toString().slice(2)}`;
}

function confidenceLabel(pred: Prediction): string {
  const maxProb = Math.max(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
  if (maxProb >= 0.68) return '🔥🔥🔥 Extreme';
  if (maxProb >= 0.60) return '🔥🔥 High';
  if (maxProb >= 0.52) return '🔥 Strong';
  if (maxProb >= 0.45) return '✅ Lean';
  return '🪙 Coin Flip';
}

function pickLabel(pred: Prediction): string {
  const outcome = getDominantOutcome(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
  if (outcome === 'home') return `${pred.home_team} Win (${pct(pred.calibrated_home_prob)})`;
  if (outcome === 'away') return `${pred.away_team} Win (${pct(pred.calibrated_away_prob)})`;
  return `Draw (${pct(pred.calibrated_draw_prob)})`;
}

function isHighConviction(pred: Prediction): boolean {
  const tier = getConfidenceTier(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
  return tier === 'extreme' || tier === 'high_conviction';
}

// ─── HTML styling ─────────────────────────────────────────────────────────────

const CSS = `
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f4f4f8; margin: 0; padding: 20px; color: #1a1a2e; }
  .container { max-width: 700px; margin: 0 auto; background: #fff;
               border-radius: 12px; overflow: hidden;
               box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
  .header { background: linear-gradient(135deg, #3d0099, #6b1fb1);
            color: #fff; padding: 28px 32px; }
  .header h1 { margin: 0 0 6px; font-size: 1.6em; }
  .header p  { margin: 0; opacity: 0.85; font-size: 0.95em; }
  .body { padding: 24px 32px; }
  .match { border: 1px solid #e8e8f0; border-radius: 8px; padding: 14px 18px;
           margin-bottom: 14px; }
  .match.hc { border-color: #6b1fb1; background: #faf7ff; }
  .match-header { font-size: 1.05em; font-weight: 700; margin-bottom: 6px; color: #1a1a2e; }
  .pick { color: #3d0099; font-weight: 600; }
  .probs { color: #555; font-size: 0.88em; margin: 4px 0; }
  .conf { font-size: 0.82em; color: #888; }
  .proj { font-size: 0.85em; color: #666; margin-top: 4px; }
  .summary-box { background: #f9f7ff; border-left: 4px solid #3d0099;
                 padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px; }
  .summary-box p { margin: 4px 0; }
  .result-ok  { color: #27ae60; font-weight: 700; }
  .result-bad { color: #e74c3c; font-weight: 700; }
  .footer { background: #f4f4f8; padding: 16px 32px; font-size: 0.78em;
            color: #888; text-align: center; }
  h2 { font-size: 1.1em; color: #3d0099; margin: 20px 0 12px; border-bottom: 1px solid #eee;
       padding-bottom: 6px; }
`;

// ─── MESSAGE 1: Gameweek Picks ────────────────────────────────────────────────

export async function sendGameweekPicksEmail(
  gameweek: number,
  season: number,
  predictions: Prediction[],
): Promise<boolean> {
  if (predictions.length === 0) {
    logger.warn({ gameweek, season }, 'No predictions — skipping picks email');
    return false;
  }

  const gwLabel = `GW${gameweek} ${getSeasonLabel(season)}`;
  const sorted = [...predictions].sort((a, b) => {
    const maxA = Math.max(a.calibrated_home_prob, a.calibrated_draw_prob, a.calibrated_away_prob);
    const maxB = Math.max(b.calibrated_home_prob, b.calibrated_draw_prob, b.calibrated_away_prob);
    return maxB - maxA;
  });

  const highConv = sorted.filter(p => isHighConviction(p));

  const matchRows = sorted.map(pred => {
    const hc = isHighConviction(pred);
    const edgeStr = pred.edge_home !== undefined && Math.abs(pred.edge_home) >= 0.04
      ? ` &nbsp;·&nbsp; Edge: ${pred.edge_home >= 0 ? '+' : ''}${pct(pred.edge_home)}`
      : '';
    return `
      <div class="match${hc ? ' hc' : ''}">
        <div class="match-header">${pred.home_team} vs ${pred.away_team} &nbsp;·&nbsp; ${pred.match_date ?? ''}</div>
        <div class="pick">⚽ ${pickLabel(pred)}</div>
        <div class="probs">🏠 ${pct(pred.calibrated_home_prob)} &nbsp; 🤝 ${pct(pred.calibrated_draw_prob)} &nbsp; ✈️ ${pct(pred.calibrated_away_prob)}${edgeStr}</div>
        <div class="proj">Proj: ${pred.most_likely_score} &nbsp;·&nbsp; O/U 2.5: ${pct(pred.over_2_5_prob)} &nbsp;·&nbsp; BTTS: ${pct(pred.btts_prob)}</div>
        <div class="conf">${confidenceLabel(pred)}</div>
      </div>`;
  }).join('\n');

  const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
  <style>${CSS}</style></head><body>
  <div class="container">
    <div class="header">
      <h1>⚽ EPL Oracle — ${gwLabel} Picks</h1>
      <p>${predictions.length} matches &nbsp;·&nbsp; ${highConv.length} high-conviction pick${highConv.length !== 1 ? 's' : ''}</p>
    </div>
    <div class="body">
      <h2>All Matches</h2>
      ${matchRows}
    </div>
    <div class="footer">
      🔥🔥🔥 Extreme &nbsp;·&nbsp; 🔥🔥 High &nbsp;·&nbsp; 🔥 Strong &nbsp;·&nbsp; ✅ Lean &nbsp;·&nbsp; 🪙 Coin Flip<br>
      EPL Oracle v4.1 &nbsp;·&nbsp; Model picks for entertainment only.
    </div>
  </div>
  </body></html>`;

  return sendEmail(`⚽ EPL Oracle — ${gwLabel} Picks`, html);
}

// ─── MESSAGE 2: Gameweek Recap ────────────────────────────────────────────────

export async function sendGameweekRecapEmail(
  gameweek: number,
  season: number,
  games: MatchWithResult[],
  metrics: RecapMetrics,
  seasonTotals?: SeasonTotals,
): Promise<boolean> {
  const gwLabel = `GW${gameweek} ${getSeasonLabel(season)}`;

  if (games.length === 0) {
    return sendEmail(
      `📊 EPL Oracle — ${gwLabel} Recap`,
      `<html><body><p>No completed match results found for ${gwLabel}. Results may still be processing.</p></body></html>`,
    );
  }

  const correct = games.filter(g => g.prediction.correct).length;
  const total = games.length;
  const accPct = (correct / total * 100).toFixed(0);
  const accEmoji = Number(accPct) >= 55 ? '🟢' : Number(accPct) >= 40 ? '🟡' : '🔴';

  const hcGames = games.filter(g => isHighConviction(g.prediction));
  const hcCorrect = hcGames.filter(g => g.prediction.correct).length;

  let seasonLine = '';
  if (seasonTotals && seasonTotals.totalMatches > 0) {
    const sAcc = (seasonTotals.accuracy * 100).toFixed(0);
    seasonLine = `<p><strong>📅 ${getSeasonLabel(season)}: ${seasonTotals.totalCorrect}/${seasonTotals.totalMatches} (${sAcc}%)</strong></p>`;
  }

  const resultRows = games.map(({ prediction: pred, homeScore, awayScore }) => {
    const dominant = getDominantOutcome(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
    const actual = homeScore > awayScore ? 'home' : homeScore < awayScore ? 'away' : 'draw';
    const ok = pred.correct;
    const badge = ok
      ? '<span class="result-ok">✅</span>'
      : '<span class="result-bad">❌</span>';
    const hcBadge = isHighConviction(pred) ? ' ⭐' : '';
    const pickStr = dominant === 'home' ? pred.home_team : dominant === 'away' ? pred.away_team : 'Draw';
    const actualStr = actual === 'home' ? pred.home_team : actual === 'away' ? pred.away_team : 'Draw';
    return `
      <div class="match${isHighConviction(pred) ? ' hc' : ''}">
        <div class="match-header">${badge}${hcBadge} ${pred.home_team} ${homeScore}–${awayScore} ${pred.away_team}</div>
        <div class="probs">Picked: <strong>${pickStr}</strong> &nbsp;·&nbsp; Actual: <strong>${actualStr}</strong></div>
        <div class="conf">${pct(Math.max(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob))} confidence</div>
      </div>`;
  }).join('\n');

  const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
  <style>${CSS}</style></head><body>
  <div class="container">
    <div class="header">
      <h1>📊 EPL Oracle — ${gwLabel} Results</h1>
      <p>${accEmoji} ${correct}/${total} correct (${accPct}%)</p>
    </div>
    <div class="body">
      <div class="summary-box">
        <p><strong>${accEmoji} This GW: ${correct}/${total} correct (${accPct}%)</strong></p>
        ${hcGames.length > 0
          ? `<p><strong>⭐ High-conviction: ${hcCorrect}/${hcGames.length} (${(hcCorrect / hcGames.length * 100).toFixed(0)}%)</strong></p>`
          : '<p><strong>⭐ No high-conviction picks this GW</strong></p>'}
        ${seasonLine}
        <p>Brier score: ${metrics.brier.toFixed(4)} (lower = better, 0.33 = random)</p>
      </div>
      <h2>Match Results</h2>
      ${resultRows}
    </div>
    <div class="footer">
      ⭐ = high-conviction pick &nbsp;·&nbsp; EPL Oracle v4.1
    </div>
  </div>
  </body></html>`;

  return sendEmail(`📊 EPL Oracle — ${gwLabel} Results`, html);
}
