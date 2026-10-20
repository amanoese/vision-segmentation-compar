import { access, mkdir, readFile, readdir, writeFile, rename } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import sharp, { type OverlayOptions } from 'sharp';
import { z } from 'zod';
import { runJevImage } from './jev.js';
import { runSamBridge } from './sam/agentBridge.js';
import { panels, pixelBoxes } from './render.js';
import type { Method, Prediction } from './types.js';
import { runSamAgent } from './agent/samAgent.js';
import { agentSettingsFromEnv, type AgentSettings } from './agent/openrouter.js';

export const METHODS: Method[] = ['sam', 'sam-agent-fixed', 'jev'];
export const ALL_METHODS: Method[] = [...METHODS, 'sam-agent'];
export const LABELS = ['SAM 3.1', 'SAM Agent (SAM 3.1, fixed query)', 'JEV'];
const schema = z.object({
  input: z.string().min(1),
  runDir: z.string().min(1),
  query: z.string().trim().min(1).max(2000),
  samPrompt: z.string().trim().min(1).max(180).optional(),
  samAgentPrompt: z.string().trim().min(1).max(180),
  samAgentMode: z.enum(['fixed', 'openrouter']).optional(),
  concurrency: z.number().int().min(1).max(4).default(3),
});
export type Config = z.infer<typeof schema> & { agentSettings?: AgentSettings };
export const methodsForConfig = (c: Config): Method[] =>
  c.samAgentMode === 'openrouter' ? ['sam', 'sam-agent', 'jev'] : METHODS;
export const labelsForConfig = (c: Config): string[] =>
  c.samAgentMode === 'openrouter'
    ? ['SAM 3.1', 'SAM Agent (SAM 3.1, max 3 LLM calls)', 'JEV']
    : LABELS;
export const agentSettingsForConfig = (c: Config) => c.agentSettings ?? agentSettingsFromEnv();
export function assertPrediction(c: Config, method: Method, d: Prediction, frame: number) {
  if (d.frame !== frame || d.query !== queryForMethod(c, method))
    throw new Error('Detection conditions mismatch');
  if (
    method === 'sam-agent' &&
    (!d.agent ||
      JSON.stringify(d.agent.settings) !== JSON.stringify(agentSettingsForConfig(c)) ||
      !Number.isInteger(d.agent.llmCalls) ||
      d.agent.llmCalls < 1 ||
      d.agent.llmCalls > 3 ||
      !['detected', 'no-detections', 'incomplete'].includes(d.status) ||
      (d.status === 'incomplete' &&
        d.boxes.length > 0 &&
        d.agent.boxSource !== 'unreviewed-candidates'))
  )
    throw new Error('Cached agent conditions mismatch');
}
const queryForMethod = (c: Config, method: Method) =>
  method === 'sam-agent-fixed'
    ? c.samAgentPrompt
    : method === 'sam'
      ? (c.samPrompt ?? c.query)
      : c.query;
export const loadJson = async (path: string) => JSON.parse(await readFile(path, 'utf8'));
export const exists = (path: string) =>
  access(path).then(
    () => true,
    () => false,
  );
export async function saveJson(path: string, value: unknown) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path + '.tmp', JSON.stringify(value, null, 2) + '\n');
  await rename(path + '.tmp', path);
}
export async function loadConfig(path: string): Promise<Config> {
  const config = schema.parse(await loadJson(path));
  return {
    ...config,
    ...(config.samAgentMode === 'openrouter' ? { agentSettings: agentSettingsFromEnv() } : {}),
    input: resolve(dirname(path), config.input),
    runDir: resolve(dirname(path), config.runDir),
  };
}
const framesDir = (c: Config) => join(c.runDir, 'source-frames');
const predictionPath = (c: Config, m: Method, f: string) =>
  join(c.runDir, 'detections', m, f.replace('.png', '.json'));
const files = async (c: Config) =>
  (await readdir(framesDir(c))).filter((f) => /^\d{6}\.png$/.test(f)).sort();
const fpsNumber = (fps: string) => {
  const [n, d] = fps.split('/').map(Number);
  if (!n || !d) throw new Error('Invalid fps');
  return n / d;
};

export async function prepare(c: Config) {
  const sha256 = createHash('sha256')
    .update(await readFile(c.input))
    .digest('hex');
  const probed = JSON.parse(
    execFileSync(
      'ffprobe',
      [
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames',
        '-of',
        'json',
        c.input,
      ],
      { encoding: 'utf8' },
    ),
  );
  const stream = probed.streams[0];
  if (!stream?.nb_frames || stream.avg_frame_rate !== stream.r_frame_rate)
    throw new Error('PoC requires a constant-frame-rate video with a known frame count');
  const protocol = {
    inputSha256: sha256,
    query: c.query,
    ...(c.samPrompt ? { samPrompt: c.samPrompt } : {}),
    samAgentPrompt: c.samAgentPrompt,
    ...(c.samAgentMode === 'openrouter'
      ? { samAgentMode: c.samAgentMode, agentSettings: agentSettingsForConfig(c) }
      : {}),
    width: stream.width,
    height: stream.height,
    fps: stream.avg_frame_rate,
    frameCount: Number(stream.nb_frames),
    models: {
      sam: 'fal-ai/sam-3-1/image',
      samAgent:
        c.samAgentMode === 'openrouter'
          ? 'SAM 3.1 with per-frame OpenRouter visual agent; at most 3 LLM requests'
          : 'SAM 3.1 with frozen Agent-derived query; no per-frame LLM',
      jev: 'jev-1.14.0',
    },
    thresholds: { jevPresence: 0.8, jevBox: 0.6 },
    layout: labelsForConfig(c),
  };
  const path = join(c.runDir, 'protocol.json');
  if (await exists(path)) {
    const old = await loadJson(path);
    if (JSON.stringify(old) !== JSON.stringify(protocol))
      throw new Error('Input or experiment conditions changed; use a new runDir');
  }
  await mkdir(framesDir(c), { recursive: true });
  const current = await files(c);
  if (current.length !== protocol.frameCount) {
    execFileSync('ffmpeg', [
      '-hide_banner',
      '-loglevel',
      'error',
      '-y',
      '-i',
      c.input,
      '-map',
      '0:v:0',
      '-fps_mode',
      'passthrough',
      join(framesDir(c), '%06d.png'),
    ]);
  }
  if ((await files(c)).length !== protocol.frameCount)
    throw new Error('Extracted frame count mismatch');
  await saveJson(path, protocol);
  return protocol;
}

export async function detect(c: Config, method: Method) {
  if (method === 'sam-agent' && c.samAgentMode !== 'openrouter')
    throw new Error('sam-agent requires samAgentMode=openrouter in config');
  const protocol = await prepare(c),
    images = await files(c);
  let next = 0,
    completed = 0,
    stopped = false;
  async function worker() {
    while (!stopped && next < images.length) {
      const file = images[next++]!,
        frame = Number(file.slice(0, 6)) - 1,
        path = predictionPath(c, method, file);
      const detectorQuery = queryForMethod(c, method);
      if (await exists(path)) {
        const cached = await loadJson(path);
        assertPrediction(c, method, cached, frame);
      } else {
        const detectionStarted = performance.now();
        const image = join(framesDir(c), file),
          raw = join(c.runDir, 'raw', method, file.slice(0, 6));
        let prediction: Prediction;
        let timingSource: 'fresh' | 'resumed' = 'fresh';
        if (method === 'sam-agent') {
          const result = await runSamAgent({
            image,
            out: raw,
            query: detectorQuery,
            settings: agentSettingsForConfig(c),
          });
          const { status, boxes, scores, ...agent } = result;
          prediction = {
            frame,
            timeSeconds: frame / fpsNumber(protocol.fps),
            query: detectorQuery,
            status,
            boxes,
            scores,
            agent,
          };
        } else if (method === 'jev') {
          const result = await runJevImage({ image, runDir: raw, target: detectorQuery });
          if (!('bbox' in result)) throw new Error('Unexpected dry-run result');
          prediction = {
            frame,
            timeSeconds: frame / fpsNumber(protocol.fps),
            query: detectorQuery,
            status: result.status,
            boxes: result.bbox ? [result.bbox] : [],
            scores: result.bbox ? [result.boxConfidence] : [],
            presenceProbability: result.presenceProbability,
            boxConfidence: result.boxConfidence,
          };
        } else {
          await mkdir(dirname(raw), { recursive: true });
          const result = await runSamBridge({
            image,
            out: raw,
            prompt: detectorQuery,
            resumeSaved: await exists(raw),
          });
          if (result.recoveredFromSavedResponse) timingSource = 'resumed';
          const boxes = pixelBoxes(result.pred_boxes, result.orig_img_w, result.orig_img_h);
          prediction = {
            frame,
            timeSeconds: frame / fpsNumber(protocol.fps),
            query: detectorQuery,
            status: boxes.length ? 'detected' : 'no-detections',
            boxes,
            scores: result.pred_scores,
          };
        }
        prediction.elapsedMs =
          prediction.agent?.elapsedMs ?? Math.round(performance.now() - detectionStarted);
        prediction.timingSource = prediction.agent?.timingSource ?? timingSource;
        await saveJson(path, prediction);
      }
      completed++;
      if (completed % 10 === 0 || completed === images.length)
        console.log(`${method}: ${completed}/${images.length}`);
    }
  }
  const results = await Promise.allSettled(
    Array.from({ length: c.concurrency }, () =>
      worker().catch((error) => {
        stopped = true;
        throw error;
      }),
    ),
  );
  const failed = results.find((r) => r.status === 'rejected');
  if (failed?.status === 'rejected') throw failed.reason;
  // Historical caches without a measurement stay blank; cache lookup latency
  // must not be presented as original inference time.
  const timingRows = await Promise.all(
    images.map(async (file) => {
      const d = (await loadJson(predictionPath(c, method, file))) as Prediction;
      return [
        d.frame,
        d.timeSeconds,
        d.status,
        d.agent?.boxSource ?? '',
        d.elapsedMs ?? '',
        d.timingSource ?? '',
      ].join(',');
    }),
  );
  await writeFile(
    join(c.runDir, 'detections', method, 'timings.csv'),
    'frame,timeSeconds,status,boxSource,elapsedMs,timingSource\n' + timingRows.join('\n') + '\n',
  );
}

export async function render(c: Config) {
  const protocol = await prepare(c),
    images = await files(c),
    out = join(c.runDir, 'comparison');
  const methods = methodsForConfig(c),
    labels = labelsForConfig(c);
  await mkdir(join(out, 'frames'), { recursive: true });
  await mkdir(join(out, 'representatives'), { recursive: true });
  const counts: Record<string, number> = Object.fromEntries(methods.map((m) => [m, 0]));
  const incompleteCounts: Record<string, number> = Object.fromEntries(methods.map((m) => [m, 0]));
  const provisionalCounts: Record<string, number> = Object.fromEntries(methods.map((m) => [m, 0]));
  const sample = new Set(
    [0, 1, 2, 3, 4, 5, 6].map((t) =>
      Math.min(images.length - 1, Math.round(t * fpsNumber(protocol.fps))),
    ),
  );
  const records: Array<unknown> = [];
  for (let i = 0; i < images.length; i++) {
    const file = images[i]!,
      source = await readFile(join(framesDir(c), file)),
      parts: OverlayOptions[] = [];
    const detections: Record<string, Prediction> = {};
    for (let col = 0; col < methods.length; col++) {
      const method = methods[col]!,
        path = predictionPath(c, method, file);
      if (!(await exists(path)))
        throw new Error(`Missing ${method} result for frame ${i}; run detection first`);
      const d = (await loadJson(path)) as Prediction;
      assertPrediction(c, method, d, i);
      detections[method] = d;
      if (d.status === 'incomplete') {
        incompleteCounts[method]!++;
        if (d.boxes.length) provisionalCounts[method]!++;
      }
      if (d.boxes.length) counts[method]!++;
      const rendered = await panels(source, d.boxes);
      for (const [row, key] of [
        [0, 'bbox'],
        [1, 'mosaic'],
      ] as const) {
        const title = `${labels[col]} | ${row === 0 ? 'bbox + 25% blue fill' : 'bbox mosaic'}`;
        const detail = `frame ${i} / ${(i / fpsNumber(protocol.fps)).toFixed(2)}s | ${d.boxes.length} box(es) | ${d.status}`;
        parts.push({
          input: await sharp(rendered[key]).resize(640, 360).png().toBuffer(),
          left: col * 640,
          top: row * 400 + 40,
        });
        parts.push({
          input: Buffer.from(
            `<svg width="640" height="40"><rect width="640" height="40" fill="#151b25"/><text x="10" y="17" font-family="sans-serif" font-size="15" fill="white">${title}</text><text x="10" y="34" font-family="sans-serif" font-size="14" fill="#aacdff">${detail}</text></svg>`,
          ),
          left: col * 640,
          top: row * 400,
        });
        if (sample.has(i)) {
          const dir = join(out, 'representatives', file.slice(0, 6));
          await mkdir(dir, { recursive: true });
          await writeFile(join(dir, `${method}-${key}.png`), rendered[key]);
        }
      }
    }
    const combined = await sharp({
      create: { width: 1920, height: 800, channels: 3, background: '#151b25' },
    })
      .composite(parts)
      .png()
      .toBuffer();
    await writeFile(join(out, 'frames', file), combined);
    if (sample.has(i)) await writeFile(join(out, 'representatives', file), combined);
    records.push({ frame: i, methods: detections });
    if ((i + 1) % 20 === 0 || i === images.length - 1)
      console.log(`render: ${i + 1}/${images.length}`);
  }
  execFileSync('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-framerate',
    protocol.fps,
    '-i',
    join(out, 'frames', '%06d.png'),
    '-c:v',
    'libx264',
    '-crf',
    '18',
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+faststart',
    join(out, 'comparison.mp4'),
  ]);
  const video = JSON.parse(
    execFileSync(
      'ffprobe',
      [
        '-v',
        'error',
        '-show_entries',
        'format=duration:stream=width,height,nb_frames,avg_frame_rate',
        '-of',
        'json',
        join(out, 'comparison.mp4'),
      ],
      { encoding: 'utf8' },
    ),
  );
  if (
    Number(video.streams[0].nb_frames) !== images.length ||
    video.streams[0].avg_frame_rate !== protocol.fps
  )
    throw new Error('Output frame count or fps mismatch');
  await saveJson(join(out, 'summary.json'), {
    frames: images.length,
    complete: true,
    detectedFrames: counts,
    incompleteFrames: incompleteCounts,
    provisionalBoxFrames: provisionalCounts,
    allDecisionsComplete: Object.values(incompleteCounts).every((count) => count === 0),
    columns: labels,
    rows: ['bbox + 25% blue fill', 'bbox mosaic'],
    samAgentPrompt: c.samAgentPrompt,
    ...(c.samAgentMode === 'openrouter' ? { renderLlmCalls: 0 } : { newLlmCalls: 0 }),
    video,
  });
  await writeFile(
    join(out, 'detections.jsonl'),
    records.map((r) => JSON.stringify(r)).join('\n') + '\n',
  );
  return { output: join(out, 'comparison.mp4'), detectedFrames: counts };
}

// One original-resolution bbox video per method. No mosaic is generated.
export async function renderVideos(c: Config) {
  const protocol = await prepare(c),
    images = await files(c);
  const out = join(c.runDir, 'bbox-videos');
  const summaries: Record<string, unknown> = {};
  const methods = methodsForConfig(c);
  for (const method of methods) {
    const frames = join(out, method, 'frames');
    const representatives = join(out, method, 'representatives');
    await mkdir(frames, { recursive: true });
    await mkdir(representatives, { recursive: true });
    let detected = 0,
      incomplete = 0,
      provisional = 0;
    const sample = new Set(
      [0, 2, 3, 5, 6].map((t) =>
        Math.min(images.length - 1, Math.round(t * fpsNumber(protocol.fps))),
      ),
    );
    for (let i = 0; i < images.length; i++) {
      const file = images[i]!,
        path = predictionPath(c, method, file);
      if (!(await exists(path))) throw new Error(`Missing ${method} result for frame ${i}`);
      const d = (await loadJson(path)) as Prediction;
      assertPrediction(c, method, d, i);
      const source = await readFile(join(framesDir(c), file));
      let { bbox } = await panels(source, d.boxes, false);
      if (d.status === 'incomplete')
        bbox = await sharp(bbox)
          .composite([
            {
              input: Buffer.from(
                `<svg width="${protocol.width}" height="40"><rect width="100%" height="40" fill="#7a3600"/><text x="12" y="28" font-family="sans-serif" font-size="24" fill="white">AGENT INCOMPLETE</text></svg>`,
              ),
              left: 0,
              top: 0,
            },
          ])
          .png()
          .toBuffer();
      await writeFile(join(frames, file), bbox);
      if (sample.has(i)) await writeFile(join(representatives, file), bbox);
      if (d.boxes.length) detected++;
      if (d.status === 'incomplete') {
        incomplete++;
        if (d.boxes.length) provisional++;
      }
    }
    const output = join(out, `${method}.mp4`);
    execFileSync('ffmpeg', [
      '-hide_banner',
      '-loglevel',
      'error',
      '-y',
      '-framerate',
      protocol.fps,
      '-i',
      join(frames, '%06d.png'),
      '-c:v',
      'libx264',
      '-crf',
      '18',
      '-pix_fmt',
      'yuv420p',
      '-movflags',
      '+faststart',
      output,
    ]);
    const video = JSON.parse(
      execFileSync(
        'ffprobe',
        [
          '-v',
          'error',
          '-show_entries',
          'format=duration:stream=width,height,nb_frames,avg_frame_rate',
          '-of',
          'json',
          output,
        ],
        { encoding: 'utf8' },
      ),
    );
    if (
      Number(video.streams[0].nb_frames) !== images.length ||
      video.streams[0].avg_frame_rate !== protocol.fps ||
      video.streams[0].width !== protocol.width ||
      video.streams[0].height !== protocol.height
    )
      throw new Error('Output timing or dimensions mismatch');
    summaries[method] = {
      query: queryForMethod(c, method),
      detectedFrames: detected,
      incompleteFrames: incomplete,
      provisionalBoxFrames: provisional,
      output,
      video,
    };
    console.log(`${method}: bbox video ready, ${detected}/${images.length} detected frames`);
  }
  await saveJson(join(out, 'summary.json'), {
    frames: images.length,
    mosaic: false,
    fillOpacity: 0.25,
    ...(c.samAgentMode === 'openrouter' ? { renderLlmCalls: 0 } : { newLlmCalls: 0 }),
    methods: summaries,
  });
  return summaries;
}
