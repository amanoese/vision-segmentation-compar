import { z } from 'zod';
import sharp from 'sharp';
import type { Box } from '../types.js';

const coords = z.tuple([z.number(), z.number(), z.number(), z.number()]);
const score = z.number().min(0).max(1);
const schema = z.object({
  masks: z.array(z.object({ url: z.string().min(1) })),
  metadata: z
    .array(
      z.object({
        index: z.number().int().nonnegative(),
        box: coords.optional(),
        score: score.optional(),
      }),
    )
    .nullish(),
  boxes: z.array(coords).nullish(),
  scores: z.array(score).nullish(),
});

export function normalizedBox(values: readonly number[], width: number, height: number): Box {
  const parsed = coords.safeParse(values);
  if (
    !parsed.success ||
    !Number.isInteger(width) ||
    width <= 0 ||
    !Number.isInteger(height) ||
    height <= 0
  )
    throw new Error('invalid-box');
  const [cx, cy, w, h] = parsed.data;
  const bounds = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  if (!(w > 0 && h > 0) || bounds.some((v) => !Number.isFinite(v) || v < -1e-6 || v > 1 + 1e-6))
    throw new Error('invalid-box');
  return {
    x0: Math.max(0, Math.floor(bounds[0]! * width)),
    y0: Math.max(0, Math.floor(bounds[1]! * height)),
    x1: Math.min(width, Math.ceil(bounds[2]! * width)),
    y1: Math.min(height, Math.ceil(bounds[3]! * height)),
  };
}

export function parseOutput(value: unknown) {
  const output = schema.parse(value);
  const indices = (output.metadata ?? []).map((m) => m.index);
  // Some fal outputs sort masks/boxes/scores and metadata by score while
  // metadata.index retains the original candidate ID. Confirm the positional
  // contract from redundant boxes/scores before accepting that representation.
  const positional =
    output.metadata?.length === output.masks.length &&
    !!output.boxes &&
    output.metadata.every(
      (meta, i) =>
        !!meta.box &&
        !!output.boxes?.[i] &&
        meta.box.every((v, j) => Math.abs(v - output.boxes![i]![j]!) <= 1e-6) &&
        (meta.score === undefined ||
          output.scores?.[i] === undefined ||
          Math.abs(meta.score - output.scores[i]!) <= 1e-6),
    );
  if (
    new Set(indices).size !== indices.length ||
    (!positional && indices.some((i) => i >= output.masks.length))
  )
    throw new Error('invalid-mask-indices');
  for (const list of [output.boxes, output.scores])
    if (list && list.length !== output.masks.length) throw new Error('invalid-mask-array-length');
  return {
    masks: output.masks.map((mask, index) => {
      const meta = positional
        ? output.metadata?.[index]
        : output.metadata?.find((m) => m.index === index);
      const normalized = meta?.box ?? output.boxes?.[index];
      const confidence = meta?.score ?? output.scores?.[index];
      if (
        meta?.box &&
        output.boxes?.[index] &&
        meta.box.some((v, i) => Math.abs(v - output.boxes![index]![i]!) > 1e-6)
      )
        throw new Error('conflicting-mask-boxes');
      return {
        index,
        url: mask.url,
        ...(normalized ? { normalized } : {}),
        ...(confidence !== undefined ? { score: confidence } : {}),
      };
    }),
  };
}

export async function maskPixels(png: Buffer, width: number, height: number) {
  const { data, info } = await sharp(png).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  if (info.width !== width || info.height !== height) throw new Error('mask-dimension-mismatch');
  const alphaMask = data.some((v, i) => i % 4 === 3 && v === 0);
  const foreground = Buffer.alloc(width * height);
  let x0 = width,
    y0 = height,
    x1 = 0,
    y1 = 0,
    foregroundPixels = 0;
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4]!,
      g = data[i * 4 + 1]!,
      b = data[i * 4 + 2]!,
      a = data[i * 4 + 3]!;
    if (
      alphaMask ? a !== 0 && a !== 255 : a !== 255 || r !== g || g !== b || (r !== 0 && r !== 255)
    )
      throw new Error('non-binary-mask');
    const present = alphaMask ? a === 255 : r === 255;
    if (present) {
      foreground[i] = 255;
      foregroundPixels++;
      const x = i % width,
        y = Math.floor(i / width);
      x0 = Math.min(x0, x);
      y0 = Math.min(y0, y);
      x1 = Math.max(x1, x + 1);
      y1 = Math.max(y1, y + 1);
    }
  }
  return {
    foreground,
    foregroundPixels,
    encoding: alphaMask ? 'binary-alpha' : 'binary-grayscale',
    box: foregroundPixels ? { x0, y0, x1, y1 } : null,
  };
}
