import { mkdtemp, readFile, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import sharp from 'sharp';
import { describe, it, expect } from 'vitest';
import { prepareImage, interpretResponse, runJevImage, MODEL } from '../src/jev.js';

const response = (
  noul = 0.95,
  confidence = 0.9,
  box = { left: 0.25, right: 0.75, top: 0.25, bottom: 0.75 },
) => ({
  model: MODEL,
  answers: { present: { type: 'noul', noul }, location: { type: 'box', confidence, box } },
  usage: { input_tokens: 123 },
});
const image = (width: number, height: number) =>
  sharp({ create: { width, height, channels: 3, background: 'white' } })
    .png()
    .toBuffer();

describe('JEV image coordinates and gating', () => {
  it('applies EXIF orientation before establishing original coordinates', async () => {
    const input = await sharp(await image(400, 300))
      .jpeg()
      .withMetadata({ orientation: 6 })
      .toBuffer();
    const p = await prepareImage(input);
    expect([p.width, p.height]).toEqual([300, 400]);
    expect((await sharp(p.jpeg).metadata()).orientation).toBeUndefined();
  });
  it('undoes resizing and asymmetric padding on a narrow image', async () => {
    const p = await prepareImage(await image(2000, 100));
    expect([p.preparedWidth, p.preparedHeight, p.top]).toEqual([1568, 280, 101]);
    const r = interpretResponse(
      response(0.95, 0.9, { left: 0.25, right: 0.75, top: 101 / 280, bottom: 179 / 280 }),
      p,
      0.8,
      0.6,
    );
    expect(r.bbox).toEqual({ x0: 500, x1: 1500, y0: 0, y1: 100 });
  });
  it('rejects absent, uncertain, degenerate, out-of-range and padding-only detections', async () => {
    const p = await prepareImage(await image(400, 100));
    expect(interpretResponse(response(0.05), p, 0.8, 0.6)).toMatchObject({
      status: 'absent',
      bbox: null,
    });
    expect(interpretResponse(response(0.5), p, 0.8, 0.6).bbox).toBeNull();
    expect(interpretResponse(response(0.9, 0.4), p, 0.8, 0.6).status).toBe('uncertain-box');
    expect(
      interpretResponse(
        response(0.9, 0.9, { left: 0.5, right: 0.5, top: 0, bottom: 1 }),
        p,
        0.8,
        0.6,
      ).status,
    ).toBe('invalid-box');
    expect(
      interpretResponse(response(0.9, 0.9, { left: 0, right: 1, top: 0, bottom: 0.1 }), p, 0.8, 0.6)
        .status,
    ).toBe('invalid-box');
    expect(() => interpretResponse(response(1.1), p, 0.8, 0.6)).toThrow();
  });
});

describe('JEV image run', () => {
  it('sends the native API contract and saves reviewable artifacts without credentials', async () => {
    const root = await mkdtemp(join(tmpdir(), 'jev-test-'));
    const source = join(root, 'source.png');
    await sharp(await image(400, 300)).toFile(source);
    let calls = 0;
    const fetcher: typeof fetch = async (url, init) => {
      calls++;
      expect(url).toBe('https://api.typesafe.ai/v1/systemone');
      expect(init?.headers).toMatchObject({ 'TypeSafe-Danger-Zone-Images': 'true' });
      const body = JSON.parse(init!.body as string);
      expect(body.model).toBe(MODEL);
      expect(body.assets[body.state.photo].format).toBe('jpg');
      expect(body.questions.location).not.toHaveProperty('criteria');
      return Response.json(response());
    };
    const runDir = join(root, 'runs', 'positive');
    expect(
      await runJevImage({ image: source, runDir }, { apiKey: 'test-secret', fetcher }),
    ).toMatchObject({ status: 'detected' });
    for (const name of [
      'result.json',
      'annotated.png',
      'crop.png',
      'response.json',
      'prepared.jpg',
    ])
      await access(join(runDir, name));
    expect(await readFile(join(runDir, 'protocol.json'), 'utf8')).not.toContain('test-secret');
    await expect(
      runJevImage({ image: source, runDir }, { apiKey: 'test-secret', fetcher }),
    ).rejects.toThrow();
    expect(calls).toBe(1);
    const negative = join(root, 'negative');
    await runJevImage(
      { image: source, runDir: negative },
      { apiKey: 'test', fetcher: async () => Response.json(response(0.01)) },
    );
    await expect(access(join(negative, 'crop.png'))).rejects.toThrow();
    await runJevImage(
      { image: source, runDir: join(root, 'dry'), dryRun: true },
      {
        fetcher: async () => {
          throw new Error('must not call');
        },
      },
    );
  });
  it('records API errors and never retries', async () => {
    const root = await mkdtemp(join(tmpdir(), 'jev-error-'));
    const source = join(root, 'source.png');
    await sharp(await image(280, 280)).toFile(source);
    let calls = 0;
    const runDir = join(root, 'run');
    await expect(
      runJevImage(
        { image: source, runDir },
        {
          apiKey: 'test',
          fetcher: async () => {
            calls++;
            return new Response('', { status: 429 });
          },
        },
      ),
    ).rejects.toThrow('JEV HTTP 429');
    expect(calls).toBe(1);
    expect(JSON.parse(await readFile(join(runDir, 'error.json'), 'utf8')).status).toBe('failed');
  });
});
