// EPL Oracle v4.1 — CLI Entry Point
// Usage:
//   npm start                             → predictions for current EPL gameweek
//   npm start -- --gw 28                 → specific gameweek (current season)
//   npm start -- --alert picks           → send GW picks to Discord
//   npm start -- --alert recap           → send last GW recap to Discord
//   npm start -- --help                  → show help

import 'dotenv/config';
import { logger } from './logger.js';
import { runPipeline } from './pipeline.js';
import { closeDb, initDb, getPredictionsByGameweek } from './db/database.js';
import { getCurrentGameweekInfo, isEPLSeason, getCurrentEPLSeason } from './api/eplClient.js';
import type { PipelineOptions } from './types.js';

type AlertMode = 'picks' | 'recap' | null;

function parseArgs(): PipelineOptions & { help: boolean; alertMode: AlertMode } {
  const args = process.argv.slice(2);
  const opts: PipelineOptions & { help: boolean; alertMode: AlertMode } = {
    help: false,
    verbose: true,
    forceRefresh: false,
    alertMode: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--help': case '-h': opts.help = true; break;
      case '--gw': case '-g': case '--week': case '-w': opts.gameweek = Number(args[++i]); break;
      case '--season': case '-s': opts.season = Number(args[++i]); break;
      case '--force-refresh': case '-f': opts.forceRefresh = true; break;
      case '--quiet': case '-q': opts.verbose = false; break;
      case '--alert': case '-a': {
        const mode = args[++i];
        if (mode === 'picks' || mode === 'recap') opts.alertMode = mode;
        else { console.error(`Unknown alert mode: "${mode}". Use "picks" or "recap".`); process.exit(1); }
        break;
      }
    }
  }
  return opts;
}

function printHelp(): void {
  console.log(`
EPL Oracle v4.1 — Poisson Prediction Engine
============================================

USAGE:
  npm start [options]

OPTIONS:
  --gw, -g N              Run predictions for specific gameweek (default: current)
  --season, -s YYYY       EPL season start year (e.g. 2025 = 2025-26 season)
  --force-refresh, -f     Bypass cache and re-fetch all data
  --quiet, -q             Suppress table output
  --alert, -a picks|recap Send Discord alert
  --help, -h              Show this help

EXAMPLES:
  npm start                              # Current GW picks
  npm start -- --gw 28                  # GW28 predictions
  npm run alerts:picks                   # Send GW picks to Discord
  npm run alerts:recap                   # Send last GW recap to Discord
  npm run train                          # Train the ML model (Python)
  npm run build-dataset                  # Build training CSV (Python)

ENVIRONMENT (.env):
  DISCORD_WEBHOOK_URL    Discord webhook URL (required for alerts)
  ODDS_API_KEY           The Odds API key (optional — better market lines)
  LOG_LEVEL              Logging level (default: info)

ARCHITECTURE:
  ESPN API → Feature Engineering (22 features) → Poisson Monte Carlo (50k sims)
  → Dixon-Coles correction → ML Meta-model → Isotonic Calibration
  → Edge Detection → SQLite → Discord
`);
}

// ─── Alert handlers ───────────────────────────────────────────────────────────

async function runPicksAlert(gw: number, season: number): Promise<void> {
  const { sendGameweekPicks } = await import('./alerts/discord.js');
  await initDb();

  let predictions = getPredictionsByGameweek(gw, season);

  if (predictions.length === 0) {
    logger.info({ gw, season }, 'No predictions in DB — running pipeline first');
    predictions = await runPipeline({ gameweek: gw, season, verbose: false });
  }

  if (predictions.length === 0) {
    logger.warn({ gw, season }, 'No matches found — nothing to send');
    return;
  }

  await sendGameweekPicks(gw, season, predictions);
}

async function runRecapAlert(gw: number, season: number): Promise<void> {
  const { sendGameweekRecap } = await import('./alerts/discord.js');
  const { processGameweekResults } = await import('./alerts/results.js');

  const recapGw = gw > 1 ? gw - 1 : 1;
  const { games, metrics } = await processGameweekResults(recapGw, season);
  await sendGameweekRecap(recapGw, season, games, metrics, metrics.seasonTotals);

  if (gw > 38) {
    const { sendSeasonSummary } = await import('./alerts/discord.js');
    logger.info('Season complete — sending summary');
    await sendSeasonSummary(season, 0, 0, 0, 0);
  }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const opts = parseArgs();

  if (opts.help) { printHelp(); process.exit(0); }

  if (!isEPLSeason() && !opts.gameweek) {
    logger.info('EPL is currently off-season. Use --gw to force a specific gameweek.');
    if (!opts.alertMode) { process.exit(0); }
  }

  const currentInfo = await getCurrentGameweekInfo();
  const gw = opts.gameweek ?? currentInfo.gameweek;
  const season = opts.season ?? currentInfo.season ?? getCurrentEPLSeason();

  logger.info({ gw, season, alert: opts.alertMode ?? 'pipeline' }, 'EPL Oracle starting');

  try {
    if (opts.alertMode === 'picks') {
      await runPicksAlert(gw, season);
      return;
    }

    if (opts.alertMode === 'recap') {
      await runRecapAlert(gw, season);
      return;
    }

    if (opts.forceRefresh) {
      const { readdirSync, unlinkSync } = await import('fs');
      const cacheDir = process.env.CACHE_DIR ?? './cache';
      try {
        for (const file of readdirSync(cacheDir)) {
          if (file.endsWith('.json')) unlinkSync(`${cacheDir}/${file}`);
        }
        logger.info('Cache cleared');
      } catch { /* may not exist */ }
    }

    const predictions = await runPipeline({ gameweek: gw, season, verbose: opts.verbose });

    if (predictions.length === 0) {
      console.log(`\nNo upcoming matches for GW${gw}, ${season}-${(season + 1).toString().slice(2)}.\n`);
    } else {
      logger.info({ predictions: predictions.length }, 'Done');
    }

  } catch (err) {
    logger.error({ err }, 'Fatal error');
    process.exit(1);
  } finally {
    closeDb();
  }
}

process.on('unhandledRejection', reason => { logger.error({ reason }, 'Unhandled rejection'); process.exit(1); });
process.on('uncaughtException', err => { logger.error({ err }, 'Uncaught exception'); closeDb(); process.exit(1); });

main();
