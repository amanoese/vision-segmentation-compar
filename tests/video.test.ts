import { mkdtemp, mkdir, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import sharp from 'sharp';
import { expect, it, vi } from 'vitest';
import { runSamBridge } from '../src/sam/agentBridge.js';
import { runJevImage } from '../src/jev.js';
import { detect, loadConfig, prepare, render } from '../src/video.js';

vi.mock('../src/sam/agentBridge.js', () => ({
  runSamBridge: vi.fn(async () => ({
    pred_boxes: [[0.25, 0.25, 0.5, 0.5]],
    orig_img_w: 280,
    orig_img_h: 280,
    pred_scores: [0.9],
  })),
}));
vi.mock('../src/jev.js', () => ({
  runJevImage: vi.fn(async () => ({
    status: 'detected',
    bbox: { x0: 70, y0: 70, x1: 210, y1: 210 },
    boxConfidence: 0.9,
    presenceProbability: 0.95,
  })),
}));

it('runs from configurable paths, freezes the Agent query, resumes cached detections and renders all frames in the requested order', async () => {
  const root = await mkdtemp(join(tmpdir(), 'vision-compare-'));
  await mkdir(join(root, 'configs'));
  await mkdir(join(root, 'input'));
  for (const name of ['000001', '000002'])
    await sharp({ create: { width: 280, height: 280, channels: 3, background: 'white' } })
      .png()
      .toFile(join(root, 'input', name + '.png'));
  execFileSync('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-framerate',
    '24',
    '-i',
    join(root, 'input', '%06d.png'),
    '-c:v',
    'libx264',
    '-pix_fmt',
    'yuv420p',
    join(root, 'input.mp4'),
  ]);
  const configPath = join(root, 'configs', 'test.json');
  await writeFile(
    configPath,
    JSON.stringify({
      input: '../input.mp4',
      runDir: '../run',
      query: '肌露出',
      samAgentPrompt: 'exposed torso',
      concurrency: 2,
    }),
  );
  const c = await loadConfig(configPath);
  expect(c.input).toBe(join(root, 'input.mp4'));
  await prepare(c);
  await detect(c, 'sam-agent-fixed');
  expect(
    vi.mocked(runSamBridge).mock.calls.every(([options]) => options.prompt === 'exposed torso'),
  ).toBe(true);
  await expect(render(c)).rejects.toThrow('Missing sam result');
  await detect(c, 'sam');
  await detect(c, 'jev');
  const before = vi.mocked(runSamBridge).mock.calls.length;
  await detect(c, 'sam-agent-fixed');
  expect(runSamBridge).toHaveBeenCalledTimes(before);
  await render(c);
  const summary = JSON.parse(await readFile(join(c.runDir, 'comparison', 'summary.json'), 'utf8'));
  expect(summary.columns).toEqual(['SAM 3.1', 'SAM Agent (SAM 3.1, fixed query)', 'JEV']);
  expect(summary).toMatchObject({
    frames: 2,
    complete: true,
    newLlmCalls: 0,
    detectedFrames: { sam: 2, 'sam-agent-fixed': 2, jev: 2 },
  });
  expect(summary.video.streams[0]).toMatchObject({
    nb_frames: '2',
    avg_frame_rate: '24/1',
    width: 1920,
    height: 800,
  });
  expect(runJevImage).toHaveBeenCalledTimes(2);
  await expect(prepare({ ...c, query: 'different' })).rejects.toThrow('conditions changed');
});
