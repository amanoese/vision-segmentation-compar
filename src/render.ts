import sharp from 'sharp';
import type { Box } from './types.js';

export function pixelBoxes(boxes: number[][], width: number, height: number): Box[] {
  const clamp = (v: number, max: number) => Math.max(0, Math.min(max, v));
  return boxes
    .map(([x, y, w, h]) => {
      if ([x, y, w, h].some((v) => v === undefined || !Number.isFinite(v)))
        throw new Error('Invalid SAM box');
      return {
        x0: clamp(Math.floor(x! * width), width),
        y0: clamp(Math.floor(y! * height), height),
        x1: clamp(Math.ceil((x! + w!) * width), width),
        y1: clamp(Math.ceil((y! + h!) * height), height),
      };
    })
    .filter((b) => b.x0 < b.x1 && b.y0 < b.y1);
}

export async function panels(
  source: Buffer,
  boxes: Box[],
  includeMosaic = true,
  mosaicCellSize = 24,
) {
  if (!Number.isInteger(mosaicCellSize) || mosaicCellSize < 2 || mosaicCellSize > 128)
    throw new Error('Mosaic cell size must be an integer from 2 to 128');
  const { width, height } = await sharp(source).metadata();
  if (!width || !height) throw new Error('Image dimensions missing');
  for (const b of boxes)
    if (
      ![b.x0, b.x1, b.y0, b.y1].every(Number.isInteger) ||
      b.x0 < 0 ||
      b.y0 < 0 ||
      b.x1 > width ||
      b.y1 > height ||
      b.x0 >= b.x1 ||
      b.y0 >= b.y1
    )
      throw new Error('Invalid pixel box');
  const rect = (b: Box, extra: string) =>
    `<rect x="${b.x0}" y="${b.y0}" width="${b.x1 - b.x0}" height="${b.y1 - b.y0}" ${extra}/>`;
  const svg = (body: string) =>
    Buffer.from(`<svg width="${width}" height="${height}">${body}</svg>`);
  const tinted = boxes.length
    ? await sharp(source)
        .composite([
          {
            input: svg(
              boxes
                .map((b) =>
                  rect(b, 'fill="#0066ff" fill-opacity="0.25" stroke="#0066ff" stroke-width="4"'),
                )
                .join(''),
            ),
          },
        ])
        .png()
        .toBuffer()
    : source;
  let mosaic = source;
  if (includeMosaic && boxes.length) {
    const small = await sharp(source)
      .resize(Math.ceil(width / mosaicCellSize), Math.ceil(height / mosaicCellSize), {
        fit: 'fill',
      })
      .png()
      .toBuffer();
    const pixelated = await sharp(small)
      .resize(width, height, { fit: 'fill', kernel: 'nearest' })
      .png()
      .toBuffer();
    const masked = await sharp(pixelated)
      .ensureAlpha()
      .composite([
        { input: svg(boxes.map((b) => rect(b, 'fill="white"')).join('')), blend: 'dest-in' },
      ])
      .png()
      .toBuffer();
    mosaic = await sharp(source)
      .composite([{ input: masked }])
      .png()
      .toBuffer();
  }
  return { bbox: tinted, mosaic };
}
