// EPL Oracle v4.1 — Market Edge Detection
// Three-way market (1X2): compare model probabilities to Vegas implied.
// Soccer markets are sharp — edges ≥ 5% are meaningful.

import type { EdgeResult, EdgeCategory } from '../types.js';

export function computeEdge(modelProb: number, marketProb: number): EdgeResult {
  const edge = modelProb - marketProb;
  const absEdge = Math.abs(edge);

  let edgeCategory: EdgeCategory;
  if (absEdge < 0.03) edgeCategory = 'none';
  else if (absEdge < 0.05) edgeCategory = 'small';
  else if (absEdge < 0.08) edgeCategory = 'meaningful';
  else if (absEdge < 0.12) edgeCategory = 'large';
  else edgeCategory = 'extreme';

  return {
    modelProb,
    vegasProb: marketProb,
    edge,
    edgeCategory,
  };
}

// ─── Confidence tier for three-way market ─────────────────────────────────────
// Unlike NFL (two outcomes), soccer has three outcomes so max prob rarely exceeds 65%.
// Tiers calibrated for soccer probabilities.

export function getConfidenceTier(
  homeProb: number,
  drawProb: number,
  awayProb: number,
): 'uncertain' | 'lean' | 'strong' | 'high_conviction' | 'extreme' {
  const maxProb = Math.max(homeProb, drawProb, awayProb);
  if (maxProb >= 0.68) return 'extreme';
  if (maxProb >= 0.60) return 'high_conviction';
  if (maxProb >= 0.52) return 'strong';
  if (maxProb >= 0.45) return 'lean';
  return 'uncertain';
}

// Dominant outcome (the one we pick)
export function getDominantOutcome(
  homeProb: number,
  drawProb: number,
  awayProb: number,
): 'home' | 'draw' | 'away' {
  if (homeProb >= drawProb && homeProb >= awayProb) return 'home';
  if (awayProb >= homeProb && awayProb >= drawProb) return 'away';
  return 'draw';
}

export function formatEdge(result: EdgeResult): string {
  const sign = result.edge >= 0 ? '+' : '';
  return `Model: ${(result.modelProb * 100).toFixed(1)}% | Market: ${(result.vegasProb * 100).toFixed(1)}% | Edge: ${sign}${(result.edge * 100).toFixed(1)}% [${result.edgeCategory.toUpperCase()}]`;
}
