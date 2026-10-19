import sharp from 'sharp';
import { expect, it } from 'vitest';
import { panels, pixelBoxes } from '../src/render.js';

it('converts normalized xywh SAM boxes into clipped original pixel boxes', () => {
  expect(
    pixelBoxes(
      [
        [0.1, 0.2, 0.3, 0.4],
        [-0.1, 0.9, 0.3, 0.3],
        [0.5, 0.5, 0, 0],
      ],
      100,
      200,
    ),
  ).toEqual([
    { x0: 10, y0: 40, x1: 40, y1: 121 },
    { x0: 0, y0: 180, x1: 20, y1: 200 },
  ]);
  expect(() => pixelBoxes([[NaN, 0, 1, 1]], 100, 100)).toThrow();
});

it('tints the inside of each box while preserving all pixels outside it', async () => {
  const source = await sharp({
    create: { width: 96, height: 96, channels: 3, background: 'white' },
  })
    .png()
    .toBuffer();
  const p = await panels(source, [{ x0: 24, y0: 24, x1: 72, y1: 72 }]);
  const pixels = await sharp(p.bbox).removeAlpha().raw().toBuffer();
  const at = (x: number, y: number) => [...pixels.subarray((y * 96 + x) * 3, (y * 96 + x) * 3 + 3)];
  expect(at(10, 10)).toEqual([255, 255, 255]);
  expect(at(40, 40)[0]).toBeGreaterThanOrEqual(190);
  expect(at(40, 40)[0]).toBeLessThanOrEqual(193);
  expect(at(40, 40)[2]).toBe(255);
  await expect(panels(source, [{ x0: -1, y0: 0, x1: 30, y1: 30 }])).rejects.toThrow();
});

it('actually pixelates inside the box, preserves outside pixels, and leaves empty detections unchanged', async () => {
  const data = Buffer.alloc(96 * 96 * 3);
  for (let y = 0; y < 96; y++)
    for (let x = 0; x < 96; x++)
      data.fill((x * 37 + y * 11) % 256, (y * 96 + x) * 3, (y * 96 + x) * 3 + 3);
  const source = await sharp(data, { raw: { width: 96, height: 96, channels: 3 } })
    .png()
    .toBuffer();
  const p = await panels(source, [{ x0: 24, y0: 24, x1: 72, y1: 72 }]);
  const pixels = await sharp(p.mosaic).removeAlpha().raw().toBuffer();
  expect(pixels.subarray((10 * 96 + 10) * 3, (10 * 96 + 10) * 3 + 3)).toEqual(
    data.subarray((10 * 96 + 10) * 3, (10 * 96 + 10) * 3 + 3),
  );
  expect(pixels.subarray((35 * 96 + 35) * 3, (35 * 96 + 35) * 3 + 3)).not.toEqual(
    data.subarray((35 * 96 + 35) * 3, (35 * 96 + 35) * 3 + 3),
  );
  const empty = await panels(source, []);
  expect(empty.bbox).toEqual(source);
  expect(empty.mosaic).toEqual(source);
});
