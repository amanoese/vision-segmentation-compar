import type { AgentSettings } from './agent/openrouter.js';
export type Box = { x0: number; y0: number; x1: number; y1: number };
export type Method = 'sam' | 'sam-agent-fixed' | 'sam-agent' | 'jev';
export interface Prediction {
  frame: number;
  timeSeconds: number;
  query: string;
  status: string;
  boxes: Box[];
  scores: number[];
  elapsedMs?: number;
  timingSource?: 'fresh' | 'resumed';
  agent?: {
    settings: AgentSettings;
    queryHistory: string[];
    llmCalls: number;
    samCalls: number;
    elapsedMs: number;
    timingSource: 'fresh' | 'resumed';
    boxSource: 'llm-selected' | 'unreviewed-candidates' | 'none';
    finishReason: string;
    reason?: string;
  };
  presenceProbability?: number;
  boxConfidence?: number;
}
