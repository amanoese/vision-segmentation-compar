export type Box = { x0: number; y0: number; x1: number; y1: number };
export type Method = 'sam' | 'sam-agent-fixed' | 'jev';
export interface Prediction {
  frame: number;
  timeSeconds: number;
  query: string;
  status: string;
  boxes: Box[];
  scores: number[];
  presenceProbability?: number;
  boxConfidence?: number;
}
