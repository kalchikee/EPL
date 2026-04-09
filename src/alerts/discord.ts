// EPL Oracle v4.1 — Discord Webhook Alerts
// Message 1: Gameweek picks briefing (before GW starts)
// Message 2: Gameweek recap (after GW results)
// Three-outcome format: Home Win / Draw / Away Win

import fetch from 'node-fetch';
import { logger } from '../logger.js';
import { getConfidenceTier, getDominantOutcome } from '../features/marketEdge.js';
import type { Prediction } from '../types.js';
import type { SeasonTotals } from '../db/database.js';

// ─── Colors ───────────────────────────────────────────────────────────────────

const COLORS = {
  picks: 0x3d0099,        // EPL purple
  edge: 0x27ae60,         // green
  recap_good: 0x2ecc71,
  recap_bad: 0xe74c3c,
  recap_neutral: 0x95a5a6,
} as const;

// ─── Discord types ────────────────────────────────────────────────────────────

interface DiscordField { name: string; value: string; inline?: boolean; }
interface DiscordEmbed {
  title?: string; description?: string; color?: number;
  fields?: DiscordField[]; footer?: { text: string }; timestamp?: string;
}
interface DiscordPayload { content?: string; embeds: DiscordEmbed[]; }

// ─── Webhook sender ───────────────────────────────────────────────────────────

async function sendWebhook(payload: DiscordPayload): Promise<boolean> {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    logger.warn('DISCORD_WEBHOOK_URL not set — skipping Discord alert');
    return false;
  }
  try {
    const resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });
    if (!resp.ok) {
      const text = await resp.text();
      logger.error({ status: resp.status, body: text }, 'Discord webhook error');
      return false;
    }
    logger.info('Discord alert sent');
    return true;
  } catch (err) {
    logger.error({ err }, 'Failed to send Discord webhook');
    return false;
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(prob: number): string {
  return (prob * 100).toFixed(1) + '%';
}

function confidenceBar(pred: Prediction): string {
  const maxProb = Math.max(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
  if (maxProb >= 0.68) return '🔥🔥🔥';
  if (maxProb >= 0.60) return '🔥🔥';
  if (maxProb >= 0.52) return '🔥';
  if (maxProb >= 0.45) return '✅';
  return '🪙';
}

function isHighConviction(pred: Prediction): boolean {
  const tier = getConfidenceTier(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
  return tier === 'extreme' || tier === 'high_conviction';
}

function pickLabel(pred: Prediction): string {
  const outcome = getDominantOutcome(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
  if (outcome === 'home') return `🏠 ${pred.home_team} Win  (${pct(pred.calibrated_home_prob)})`;
  if (outcome === 'away') return `✈️ ${pred.away_team} Win  (${pct(pred.calibrated_away_prob)})`;
  return `🤝 Draw  (${pct(pred.calibrated_draw_prob)})`;
}

function probLine(pred: Prediction): string {
  return `🏠 ${pct(pred.calibrated_home_prob)}  🤝 ${pct(pred.calibrated_draw_prob)}  ✈️ ${pct(pred.calibrated_away_prob)}`;
}

function getSeasonLabel(season: number): string {
  return `${season}-${(season + 1).toString().slice(2)}`;
}

// ─── MESSAGE 1: Gameweek Picks Briefing ───────────────────────────────────────

export async function sendGameweekPicks(
  gameweek: number,
  season: number,
  predictions: Prediction[],
): Promise<boolean> {
  if (predictions.length === 0) {
    logger.warn({ gameweek, season }, 'No predictions — skipping picks alert');
    return false;
  }

  const seasonLabel = getSeasonLabel(season);
  const gwLabel = `GW${gameweek} ${seasonLabel}`;

  // Sort by confidence (highest max prob first)
  const sorted = [...predictions].sort((a, b) => {
    const maxA = Math.max(a.calibrated_home_prob, a.calibrated_draw_prob, a.calibrated_away_prob);
    const maxB = Math.max(b.calibrated_home_prob, b.calibrated_draw_prob, b.calibrated_away_prob);
    return maxB - maxA;
  });

  const highConv = sorted.filter(p => isHighConviction(p));
  const hasEdge = sorted.some(p =>
    (p.edge_home !== undefined && Math.abs(p.edge_home) >= 0.05) ||
    (p.edge_away !== undefined && Math.abs(p.edge_away) >= 0.05)
  );

  // ── Embed 1: All Picks ────────────────────────────────────────────────────
  const picksFields: DiscordField[] = sorted.slice(0, 20).map(pred => {
    const conf = confidenceBar(pred);
    const edgeStr = pred.edge_home !== undefined && Math.abs(pred.edge_home) >= 0.04
      ? `  Edge-H: ${pred.edge_home >= 0 ? '+' : ''}${pct(pred.edge_home)}`
      : '';
    return {
      name: `${conf} ${pred.home_team} vs ${pred.away_team}  ·  ${pred.match_date ?? ''}`,
      value: [
        `**Pick:** ${pickLabel(pred)}`,
        `${probLine(pred)}`,
        `Proj: ${pred.most_likely_score}  |  O/U 2.5: ${pct(pred.over_2_5_prob)}  |  BTTS: ${pct(pred.btts_prob)}${edgeStr}`,
      ].join('\n'),
      inline: false,
    };
  });

  const picksEmbed: DiscordEmbed = {
    title: `⚽ EPL Oracle — ${gwLabel} Picks`,
    description: `${predictions.length} matches this gameweek  ·  ${highConv.length} high-conviction pick${highConv.length !== 1 ? 's' : ''}${hasEdge ? '  ·  ⚡ edge games flagged' : ''}`,
    color: COLORS.picks,
    fields: picksFields,
    footer: { text: '🔥🔥🔥 Extreme  🔥🔥 High  🔥 Strong  ✅ Lean  🪙 Coin Flip  ·  EPL Oracle v4.1' },
    timestamp: new Date().toISOString(),
  };

  const embeds: DiscordEmbed[] = [picksEmbed];

  // ── Embed 2: High-Conviction Picks ────────────────────────────────────────
  if (highConv.length > 0) {
    const hcFields: DiscordField[] = highConv.map(pred => {
      const tier = getConfidenceTier(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
      const tierLabel = tier === 'extreme' ? '🔥🔥🔥 EXTREME' : '🔥🔥 HIGH CONVICTION';
      const outcome = getDominantOutcome(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
      const reasons: string[] = [];

      const maxProb = Math.max(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
      reasons.push(`Model confidence: ${pct(maxProb)}`);
      if (pred.edge_home !== undefined && pred.edge_home >= 0.05) reasons.push(`+${pct(pred.edge_home)} home edge vs market`);
      if (pred.edge_away !== undefined && pred.edge_away >= 0.05) reasons.push(`+${pct(pred.edge_away)} away edge vs market`);
      if (pred.feature_vector.away_euro_fatigue) reasons.push('Away team on UEFA fatigue');
      if (pred.feature_vector.rest_days_diff >= 4) reasons.push(`Home has ${pred.feature_vector.rest_days_diff}+ more rest days`);
      if (pred.feature_vector.elo_diff > 150) reasons.push(`Home Elo advantage: ${pred.feature_vector.elo_diff.toFixed(0)} pts`);

      return {
        name: `${tierLabel}: ${pred.home_team} vs ${pred.away_team}`,
        value: [
          `**Pick:** ${pickLabel(pred)}`,
          `**Why:** ${reasons.join(' · ')}`,
          `Proj: ${pred.most_likely_score}  |  Poisson λ: ${pred.expected_home_goals.toFixed(2)}–${pred.expected_away_goals.toFixed(2)}`,
        ].join('\n'),
        inline: false,
      };
    });

    embeds.push({
      title: `⭐ High-Conviction Picks — ${highConv.length} match${highConv.length !== 1 ? 'es' : ''}`,
      description: 'Matches where model max outcome probability ≥ 60%. These are the sharpest signals.',
      color: COLORS.edge,
      fields: hcFields,
      footer: { text: 'Bet responsibly. Model picks are for entertainment only.' },
    });
  }

  return sendWebhook({ embeds });
}

// ─── MESSAGE 2: Gameweek Recap ────────────────────────────────────────────────

export async function sendGameweekRecap(
  gameweek: number,
  season: number,
  games: Array<{
    prediction: Prediction;
    homeScore: number;
    awayScore: number;
  }>,
  metrics: { accuracy: number; brier: number; highConvAccuracy: number | null },
  seasonTotals?: SeasonTotals,
): Promise<boolean> {
  const gwLabel = `GW${gameweek} ${getSeasonLabel(season)}`;

  if (games.length === 0) {
    return sendWebhook({
      embeds: [{
        title: `📊 EPL Oracle — ${gwLabel} Recap`,
        description: 'No completed matches found. Results may still be processing.',
        color: COLORS.recap_neutral,
        timestamp: new Date().toISOString(),
      }],
    });
  }

  const correct = games.filter(g => g.prediction.correct).length;
  const total = games.length;
  const accPct = (correct / total) * 100;
  const color = accPct >= 55 ? COLORS.recap_good : accPct >= 40 ? COLORS.recap_neutral : COLORS.recap_bad;
  const accEmoji = accPct >= 55 ? '🟢' : accPct >= 40 ? '🟡' : '🔴';

  const hcGames = games.filter(g => isHighConviction(g.prediction));
  const hcCorrect = hcGames.filter(g => g.prediction.correct).length;

  const gameLines = games.map(({ prediction: pred, homeScore, awayScore }) => {
    const outcome = homeScore > awayScore ? 'home' : homeScore < awayScore ? 'away' : 'draw';
    const dominant = getDominantOutcome(pred.calibrated_home_prob, pred.calibrated_draw_prob, pred.calibrated_away_prob);
    const ok = pred.correct ? '✅' : '❌';
    const bet = isHighConviction(pred) ? ' ⭐' : '';
    const resultLabel = outcome === 'home' ? pred.home_team : outcome === 'away' ? pred.away_team : 'Draw';
    const pickLabel = dominant === 'home' ? pred.home_team : dominant === 'away' ? pred.away_team : 'Draw';
    return `${ok}${bet} **${pred.home_team}** ${homeScore}–${awayScore} **${pred.away_team}** *(picked ${pickLabel}, actual: ${resultLabel})*`;
  }).join('\n');

  let seasonLine = '';
  if (seasonTotals && seasonTotals.totalMatches > 0) {
    const sAcc = (seasonTotals.accuracy * 100).toFixed(0);
    const sHcAcc = seasonTotals.hcMatches > 0
      ? `  ·  ⭐ ${seasonTotals.hcCorrect}/${seasonTotals.hcMatches} (${(seasonTotals.hcAccuracy * 100).toFixed(0)}%) high-conv`
      : '';
    seasonLine = `**📅 ${getSeasonLabel(season)}: ${seasonTotals.totalCorrect}/${seasonTotals.totalMatches} (${sAcc}%)${sHcAcc}**`;
  }

  const summaryLines = [
    `**${accEmoji} This GW: ${correct}/${total} correct  (${accPct.toFixed(0)}%)**`,
    hcGames.length > 0
      ? `**⭐ High-conviction: ${hcCorrect}/${hcGames.length}  (${((hcCorrect / hcGames.length) * 100).toFixed(0)}%)**`
      : '**⭐ No high-conviction picks this GW**',
    seasonLine,
    `Brier score: ${metrics.brier.toFixed(4)} *(lower = better, 0.33 = random for 3-way)*`,
  ].filter(Boolean).join('\n');

  const embed: DiscordEmbed = {
    title: `📊 EPL Oracle — ${gwLabel} Results`,
    color,
    fields: [
      { name: '📈 Summary', value: summaryLines, inline: false },
      { name: '⚽ Match-by-Match', value: gameLines.slice(0, 1000) || 'No results.', inline: false },
    ],
    footer: { text: '⭐ = high-conviction pick  ·  EPL Oracle v4.1' },
    timestamp: new Date().toISOString(),
  };

  return sendWebhook({ embeds: [embed] });
}

// ─── Season-end summary ───────────────────────────────────────────────────────

export async function sendSeasonSummary(
  season: number,
  totalCorrect: number,
  totalMatches: number,
  hcCorrect: number,
  hcMatches: number,
): Promise<boolean> {
  const acc = totalMatches > 0 ? (totalCorrect / totalMatches * 100).toFixed(1) : '0';
  const hcAcc = hcMatches > 0 ? (hcCorrect / hcMatches * 100).toFixed(1) : 'N/A';
  const seasonLabel = getSeasonLabel(season);

  return sendWebhook({
    embeds: [{
      title: `🏆 EPL Oracle — ${seasonLabel} Season Complete`,
      description: [
        `**Overall:** ${totalCorrect}/${totalMatches} (${acc}%)`,
        `**High-conviction:** ${hcCorrect}/${hcMatches} (${hcAcc}%)`,
        '',
        'The season is over. See you in August. ⚽',
      ].join('\n'),
      color: 0x3d0099,
      timestamp: new Date().toISOString(),
      footer: { text: `EPL Oracle v4.1 · ${seasonLabel} Season Complete` },
    }],
  });
}
