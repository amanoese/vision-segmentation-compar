import { mkdtemp, mkdir, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import sharp from 'sharp';
import { expect, it, vi } from 'vitest';
import { runSamBridge } from '../src/sam/agentBridge.js';
import { runJevImage } from '../src/jev.js';
import { runSamAgent } from '../src/agent/samAgent.js';
import { detect, loadConfig, prepare, render, renderVideos } from '../src/video.js';

vi.mock('../src/sam/agentBridge.js', () => ({
  runSamBridge: vi.fn(async () => ({
    pred_boxes: [[0.25, 0.25, 0.5, 0.5]],
    orig_img_w: 280,
    orig_img_h: 280,
    pred_scores: [0.9],
  })),
}));
vi.mock('../src/agent/samAgent.js', () => ({ runSamAgent: vi.fn() }));

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
  await renderVideos(c);
  const individual = JSON.parse(
    await readFile(join(c.runDir, 'bbox-videos', 'summary.json'), 'utf8'),
  );
  expect(individual.mosaic).toBe(false);
  expect(Object.keys(individual.methods)).toEqual(['sam', 'sam-agent-fixed', 'jev']);
  for (const method of Object.values(individual.methods) as Array<{
    video: { streams: Array<Record<string, unknown>> };
  }>) {
    expect(method.video.streams[0]).toMatchObject({
      nb_frames: '2',
      avg_frame_rate: '24/1',
      width: 280,
      height: 280,
    });
  }
  expect(runJevImage).toHaveBeenCalledTimes(2);
  await expect(prepare({ ...c, query: 'different' })).rejects.toThrow('conditions changed');
  const english = {
    ...c,
    runDir: join(root, 'sam-english'),
    query: 'フィギュア',
    samPrompt: 'figurine',
  };
  await detect(english, 'sam');
  expect(
    vi
      .mocked(runSamBridge)
      .mock.calls.slice(-2)
      .every(([options]) => options.prompt === 'figurine'),
  ).toBe(true);
  await detect(english, 'jev');
  expect(
    vi
      .mocked(runJevImage)
      .mock.calls.slice(-2)
      .every(([options]) => options.target === 'フィギュア'),
  ).toBe(true);
});

it('connects per-frame OpenRouter decisions, caches model conditions, and renders incomplete frames distinctly in 2x2 outputs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'vision-agent-integration-'));
  vi.stubEnv('SAM_AGENT_MODEL', 'test/vision-model');
  try {
    const input = join(root, 'input.mp4');
    execFileSync('ffmpeg', [
      '-hide_banner',
      '-loglevel',
      'error',
      '-f',
      'lavfi',
      '-i',
      'color=c=white:s=280x280:r=24',
      '-frames:v',
      '2',
      '-c:v',
      'libx264',
      '-pix_fmt',
      'yuv420p',
      input,
    ]);
    const configPath = join(root, 'config.json');
    await writeFile(
      configPath,
      JSON.stringify({
        input,
        runDir: join(root, 'run'),
        query: 'skin exposure',
        samAgentPrompt: 'unused legacy prompt',
        samAgentMode: 'openrouter',
        concurrency: 2,
      }),
    );
    const c = await loadConfig(configPath);
    expect(c.agentSettings).toMatchObject({ model: 'test/vision-model', maxLlmCalls: 3 });
    vi.mocked(runSamAgent).mockImplementation(async (options) => ({
      status: options.image.endsWith('000001.png') ? 'incomplete' : 'detected',
      boxes: [{ x0: 70, y0: 70, x1: 210, y1: 210 }],
      scores: [0.9],
      boxSource: options.image.endsWith('000001.png') ? 'unreviewed-candidates' : 'llm-selected',
      timingSource: 'fresh',
      settings: options.settings,
      queryHistory: ['bare chest'],
      llmCalls: 3,
      samCalls: 1,
      elapsedMs: 123,
      finishReason: options.image.endsWith('000001.png') ? 'llm-call-limit' : 'selected-masks',
    }));
    await detect(c, 'sam-agent');
    expect(runSamAgent).toHaveBeenCalledTimes(2);
    expect(vi.mocked(runSamAgent).mock.calls.every(([o]) => o.query === 'skin exposure')).toBe(
      true,
    );
    const firstPrediction = await readFile(
      join(c.runDir, 'detections/sam-agent/000001.json'),
      'utf8',
    );
    expect(JSON.parse(firstPrediction)).toMatchObject({
      status: 'incomplete',
      elapsedMs: 123,
      timingSource: 'fresh',
      agent: { boxSource: 'unreviewed-candidates' },
    });
    await detect(c, 'sam-agent');
    expect(runSamAgent).toHaveBeenCalledTimes(2);
    expect(await readFile(join(c.runDir, 'detections/sam-agent/000001.json'), 'utf8')).toBe(
      firstPrediction,
    );
    const csv = await readFile(join(c.runDir, 'detections/sam-agent/timings.csv'), 'utf8');
    expect(csv).toContain('frame,timeSeconds,status,boxSource,elapsedMs,timingSource');
    expect(csv).toContain('0,0,incomplete,unreviewed-candidates,123,fresh');
    await detect(c, 'sam');
    await detect(c, 'jev');
    for (const method of ['sam', 'jev']) {
      const p = JSON.parse(
        await readFile(join(c.runDir, 'detections', method, '000001.json'), 'utf8'),
      );
      expect(p.elapsedMs).toBeGreaterThanOrEqual(0);
      expect(p.timingSource).toBe('fresh');
    }
    await render(c);
    const summary = JSON.parse(
      await readFile(join(c.runDir, 'comparison', 'summary.json'), 'utf8'),
    );
    expect(summary.columns[1]).toBe('SAM Agent (SAM 3.1, max 3 LLM calls)');
    expect(summary).toMatchObject({
      incompleteFrames: { 'sam-agent': 1 },
      allDecisionsComplete: false,
    });
    execFileSync(process.execPath, ['--import', 'tsx', 'src/render2x2.ts', configPath], {
      cwd: process.cwd(),
      stdio: 'pipe',
    });
    const grid = JSON.parse(
      await readFile(join(c.runDir, 'comparison-2x2', 'summary.json'), 'utf8'),
    );
    expect(grid).toMatchObject({
      detectedFrames: { 'sam-agent': 2 },
      provisionalBoxFrames: { 'sam-agent': 1 },
      incompleteFrames: { 'sam-agent': 1 },
      allDecisionsComplete: false,
    });
    const pixels = await sharp(
      await readFile(join(c.runDir, 'comparison-2x2/bbox/frames/000001.png')),
    )
      .removeAlpha()
      .raw()
      .toBuffer();
    const center = (890 * 1920 + 480) * 3;
    expect(pixels[center + 2]!).toBeGreaterThan(pixels[center]! + 30);
    for (const kind of ['bbox', 'mosaic']) {
      expect(grid.outputs[kind].video.streams[0]).toMatchObject({
        nb_frames: '2',
        avg_frame_rate: '24/1',
        width: 1920,
        height: 1160,
      });
    }
    // Explicit partial rendering must preserve missing frames as missing, not as negative predictions.
    const { rm } = await import('node:fs/promises');
    await rm(join(c.runDir, 'detections/sam-agent/000001.json'));
    expect(() =>
      execFileSync(process.execPath, ['--import', 'tsx', 'src/render2x2.ts', configPath], {
        cwd: process.cwd(),
        stdio: 'pipe',
      }),
    ).toThrow();
    execFileSync(
      process.execPath,
      ['--import', 'tsx', 'src/render2x2.ts', configPath, '--allow-missing'],
      { cwd: process.cwd(), stdio: 'pipe' },
    );
    const partial = JSON.parse(
      await readFile(join(c.runDir, 'comparison-2x2-partial/summary.json'), 'utf8'),
    );
    expect(partial).toMatchObject({
      partial: true,
      missingFrames: { 'sam-agent': 1 },
      detectedFrames: { 'sam-agent': 1 },
      allDecisionsComplete: false,
    });
    await expect(
      prepare({ ...c, agentSettings: { ...c.agentSettings!, model: 'other/model' } }),
    ).rejects.toThrow('conditions changed');
  } finally {
    vi.unstubAllEnvs();
    const { rm } = await import('node:fs/promises');
    await rm(root, { recursive: true, force: true });
  }
}, 30_000);
