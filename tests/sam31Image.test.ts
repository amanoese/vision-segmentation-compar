import { describe, expect, it } from 'vitest';
import sharp from 'sharp';
import { normalizedBox, parseOutput, maskPixels } from '../src/sam/response.js';

describe('SAM 3.1 image geometry and masks', () => {
  it('converts normalized cxcywh to exclusive pixel bounds', () => {
    expect(normalizedBox([0.5, 0.5, 0.5, 0.5], 1280, 720)).toEqual({
      x0: 320,
      y0: 180,
      x1: 960,
      y1: 540,
    });
  });
  it.each([
    [NaN, 0.5, 0.1, 0.1],
    [0.5, 0.5, -1, 0.1],
    [1, 0.5, 1, 0.1],
    [0.5, 0.5, 0, 0.1],
  ])('rejects invalid box %j', (...box) => {
    expect(() => normalizedBox(box, 1280, 720)).toThrow('invalid-box');
  });
  it('distinguishes no detections from a missing masks field', () => {
    expect(parseOutput({ masks: [] }).masks).toEqual([]);
    expect(
      parseOutput({ masks: [], image: null, metadata: null, scores: [], boxes: [] }).masks,
    ).toEqual([]);
    expect(() => parseOutput({})).toThrow();
  });
  it('uses metadata index rather than metadata array position', () => {
    const result = parseOutput({
      masks: [{ url: 'https://example.com/0.png' }, { url: 'https://example.com/1.png' }],
      metadata: [{ index: 1, box: [0.5, 0.5, 0.2, 0.2], score: 0.8 }],
    });
    expect(result.masks[0]?.normalized).toBeUndefined();
    expect(result.masks[1]?.normalized).toEqual([0.5, 0.5, 0.2, 0.2]);
  });
  it('rejects ambiguous metadata indices and top-level lengths', () => {
    expect(() =>
      parseOutput({ masks: [{ url: 'x' }], metadata: [{ index: 0 }, { index: 0 }] }),
    ).toThrow();
    expect(() => parseOutput({ masks: [{ url: 'x' }], boxes: [] })).toThrow();
  });
  it('derives a box from a verified binary mask', async () => {
    const png = await sharp(Buffer.from([0, 255, 0, 0, 255, 0]), {
      raw: { width: 3, height: 2, channels: 1 },
    })
      .png()
      .toBuffer();
    const mask = await maskPixels(png, 3, 2);
    expect(mask.box).toEqual({ x0: 1, y0: 0, x1: 2, y1: 2 });
    expect(mask.foregroundPixels).toBe(2);
  });
  it('does not mistake a colored image for a binary mask', async () => {
    const png = await sharp(Buffer.from([255, 0, 0, 0, 0, 0]), {
      raw: { width: 2, height: 1, channels: 3 },
    })
      .png()
      .toBuffer();
    await expect(maskPixels(png, 2, 1)).rejects.toThrow('non-binary-mask');
  });
  it('rejects masks with dimensions different from input', async () => {
    const png = await sharp({ create: { width: 2, height: 2, channels: 3, background: 'black' } })
      .png()
      .toBuffer();
    await expect(maskPixels(png, 3, 2)).rejects.toThrow('mask-dimension-mismatch');
  });
});
