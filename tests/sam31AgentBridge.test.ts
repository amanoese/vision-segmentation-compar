import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import sharp from 'sharp';
import { runSamBridge } from '../src/sam/agentBridge.js';

it('converts fal cxcywh into official Agent normalized xywh and preserves masks', async () => {
  const root = await mkdtemp(join(tmpdir(), 'sam-agent-'));
  try {
    const image = join(root, 'input.png');
    await sharp({ create: { width: 8, height: 4, channels: 3, background: 'gray' } })
      .png()
      .toFile(image);
    const mask = await sharp(
      Buffer.from([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
      ]),
      { raw: { width: 8, height: 4, channels: 1 } },
    )
      .png()
      .toBuffer();
    const result = await runSamBridge(
      { image, out: join(root, 'out'), prompt: 'skin' },
      {
        infer: async (_input, onEnqueue) => {
          await onEnqueue('stub-agent');
          return {
            requestId: 'stub-agent',
            elapsedMs: 1,
            data: {
              masks: [{ url: `data:image/png;base64,${mask.toString('base64')}` }],
              metadata: [{ index: 0, score: 0.9, box: [0.5, 0.5, 0.5, 0.5] }],
            },
          };
        },
      },
    );
    expect(result.pred_boxes).toEqual([[0.25, 0.25, 0.5, 0.5]]);
    expect(result.pred_scores).toEqual([0.9]);
    expect(await readFile(result.png_masks[0]!)).toEqual(mask);
    expect(await readFile(join(root, 'out/request.json'), 'utf8')).not.toContain('base64');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
