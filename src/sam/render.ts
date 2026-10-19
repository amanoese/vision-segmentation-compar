import sharp from 'sharp';
import type { Box } from '../types.js';

export interface Detection {
  index: number;
  box: Box | null;
  normalized?: number[];
  score?: number;
  boxSource: 'api' | 'mask-derived' | 'unresolved';
  foregroundPixels?: number;
  maskEncoding?: string;
  issue?: string;
}
const escape = (s: string) =>
  s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

export async function renderOverlay(
  source: Buffer,
  detections: Detection[],
  foreground: Buffer[],
  width: number,
  height: number,
) {
  const tint = Buffer.alloc(width * height * 4);
  for (let i = 0; i < width * height; i++)
    if (foreground.some((mask) => mask[i])) {
      tint[i * 4] = 0;
      tint[i * 4 + 1] = 210;
      tint[i * 4 + 2] = 255;
      tint[i * 4 + 3] = 85;
    }
  const boxes = detections
    .map((d) => {
      if (!d.box) return '';
      const { x0, y0, x1, y1 } = d.box;
      const label = `#${d.index + 1}${d.score !== undefined ? ` score ${d.score.toFixed(3)}` : ''}`;
      const ly = Math.max(22, y0);
      return `<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="none" stroke="#00d2ff" stroke-width="3"/><rect x="${x0}" y="${ly - 22}" width="180" height="22" fill="#10212b"/><text x="${x0 + 5}" y="${ly - 6}" font-size="15" fill="white">${escape(label)}</text>`;
    })
    .join('');
  return sharp(source)
    .composite([
      { input: tint, raw: { width, height, channels: 4 } },
      {
        input: Buffer.from(
          `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg"><g font-family="sans-serif">${boxes}</g></svg>`,
        ),
      },
    ])
    .png()
    .toBuffer();
}
