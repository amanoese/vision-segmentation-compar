import { readFile, mkdir, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import sharp from 'sharp';
import { panels } from './render.js';
import {
  loadConfig,
  prepare,
  loadJson,
  saveJson,
  methodsForConfig,
  labelsForConfig,
  assertPrediction,
} from './video.js';
import type { Prediction } from './types.js';

// Local rendering only: consumes saved detections, never calls model APIs.
const config = await loadConfig(process.argv[2] ?? 'configs/skin-sns-openrouter-agent.json');
const protocol = await prepare(config);
const allowMissing = process.argv.includes('--allow-missing');
const mosaicOnly = process.argv.includes('--mosaic-only');
const mosaicCellSize = Number(
  process.argv.find((arg) => arg.startsWith('--mosaic-cell-size='))?.split('=')[1] ?? 24,
);
if (!Number.isInteger(mosaicCellSize) || mosaicCellSize < 2 || mosaicCellSize > 128)
  throw new Error('Mosaic cell size must be an integer from 2 to 128');
const out = join(
  config.runDir,
  mosaicOnly
    ? mosaicCellSize === 24
      ? 'comparison-mosaic'
      : `comparison-mosaic-cell-${mosaicCellSize}`
    : allowMissing
      ? 'comparison-2x2-partial'
      : 'comparison-2x2',
);
const files = (await readdir(join(config.runDir, 'source-frames')))
  .filter((f) => /^\d{6}\.png$/.test(f))
  .sort();
const [fpsNumerator, fpsDenominator] = protocol.fps.split('/').map(Number);
const previewFrame = Math.min(
  files.length - 1,
  Math.round(((mosaicOnly ? 1.5 : 3) * fpsNumerator!) / fpsDenominator!),
);
const kinds: ('bbox' | 'mosaic')[] = mosaicOnly ? ['mosaic'] : ['bbox', 'mosaic'];
for (const kind of kinds) await mkdir(join(out, kind, 'frames'), { recursive: true });
const methods = methodsForConfig(config);
const methodLabels = labelsForConfig(config).map((label, index) =>
  methods[index] === 'sam-agent'
    ? `SAM Agent (SAM 3.1 + ${protocol.agentSettings!.model}, max ${protocol.agentSettings!.maxLlmCalls} LLM calls)`
    : label,
);
const xml = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const bannerHeight = mosaicOnly ? 120 : 0;
const queryLines: string[] = [];
for (const word of config.query.split(' ')) {
  const last = queryLines.length - 1;
  if (last < 0 || queryLines[last]!.length + word.length + 1 > 130) queryLines.push(word);
  else queryLines[last] += ' ' + word;
}
if (mosaicOnly && queryLines.length > 2) throw new Error('Query exceeds mosaic banner capacity');
const banner = Buffer.from(
  `<svg width="1920" height="120"><rect width="1920" height="120" fill="#0c1421"/><text x="20" y="36" font-family="sans-serif" font-size="29" fill="white">Detected bounding boxes: mosaic comparison | cell size ~${mosaicCellSize} px</text>${queryLines.map((line, i) => `<text x="20" y="${72 + i * 29}" font-family="sans-serif" font-size="22" fill="#d0d9e8">${i === 0 ? 'Query: ' : ''}${xml(line)}</text>`).join('')}</svg>`,
);
const labels = ['Original', ...methodLabels];
const header = (label: string) =>
  Buffer.from(
    `<svg width="960" height="40"><rect width="960" height="40" fill="#151b25"/><text x="16" y="29" font-family="sans-serif" font-size="24" fill="white">${xml(label)}</text></svg>`,
  );
const headers = labels.map(header);
const detected: Record<string, number> = Object.fromEntries(methods.map((m) => [m, 0]));
const provisionalCounts: Record<string, number> = Object.fromEntries(methods.map((m) => [m, 0]));
const missingCounts: Record<string, number> = Object.fromEntries(methods.map((m) => [m, 0]));
const incompleteCounts: Record<string, number> = Object.fromEntries(methods.map((m) => [m, 0]));
for (let i = 0; i < files.length; i++) {
  const file = files[i]!;
  const source = await readFile(join(config.runDir, 'source-frames', file));
  const original = await sharp(source).resize(960, 540).png().toBuffer();
  const incomplete = new Set<string>();
  const missing = new Set<string>();
  const tiles: Record<string, Buffer[]> = { bbox: [original], mosaic: [original] };
  for (const method of methods) {
    let d: Prediction;
    try {
      d = await loadJson(join(config.runDir, 'detections', method, file.replace('.png', '.json')));
    } catch (error) {
      if (
        !allowMissing ||
        method !== 'sam-agent' ||
        !(error instanceof Error) ||
        !('code' in error) ||
        error.code !== 'ENOENT'
      )
        throw error;
      missing.add(method);
      missingCounts[method]!++;
      for (const kind of kinds) tiles[kind]!.push(original);
      continue;
    }
    assertPrediction(config, method, d, i);
    if (d.status === 'incomplete') {
      incomplete.add(method);
      incompleteCounts[method]!++;
      if (d.boxes.length) provisionalCounts[method]!++;
    }
    if (d.boxes.length) detected[method]!++;
    const rendered = await panels(source, d.boxes, true, mosaicCellSize);
    for (const kind of kinds)
      tiles[kind]!.push(await sharp(rendered[kind]).resize(960, 540).png().toBuffer());
  }
  for (const kind of kinds) {
    const composite = tiles[kind]!.flatMap((input, index) => {
      const left = (index % 2) * 960,
        top = bannerHeight + Math.floor(index / 2) * 580;
      return [
        {
          input:
            index > 0 && missing.has(methods[index - 1]!)
              ? header(`${labels[index]} | NOT PROCESSED`)
              : index > 0 && incomplete.has(methods[index - 1]!)
                ? header(`${labels[index]} | INCOMPLETE`)
                : headers[index]!,
          left,
          top,
        },
        { input, left, top: top + 40 },
      ];
    });
    const frame = await sharp({
      create: { width: 1920, height: 1160 + bannerHeight, channels: 3, background: '#151b25' },
    })
      .composite([...(mosaicOnly ? [{ input: banner, left: 0, top: 0 }] : []), ...composite])
      .png()
      .toBuffer();
    await writeFile(join(out, kind, 'frames', file), frame);
    if (i === previewFrame) await writeFile(join(out, `${kind}-preview.png`), frame);
  }
  if ((i + 1) % 20 === 0 || i === files.length - 1)
    console.log(`2x2 render: ${i + 1}/${files.length}`);
}
const outputs: Record<string, unknown> = {};
for (const kind of kinds) {
  const output = join(out, `${kind}-2x2.mp4`);
  execFileSync('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-framerate',
    protocol.fps,
    '-i',
    join(out, kind, 'frames', '%06d.png'),
    '-an',
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
  const stream = video.streams[0];
  if (
    Number(stream.nb_frames) !== files.length ||
    stream.avg_frame_rate !== protocol.fps ||
    stream.width !== 1920 ||
    stream.height !== 1160 + bannerHeight
  )
    throw new Error('Output frame count, fps or dimensions mismatch');
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', output, '-f', 'null', '-']);
  outputs[kind] = { output, video };
}
await saveJson(join(out, 'summary.json'), {
  query: config.query,
  frames: files.length,
  layout: [
    ['Original', methodLabels[0]],
    [methodLabels[1], methodLabels[2]],
  ],
  detectedFrames: detected,
  incompleteFrames: incompleteCounts,
  missingFrames: missingCounts,
  partial: Object.values(missingCounts).some((count) => count > 0),
  provisionalBoxFrames: provisionalCounts,
  allDecisionsComplete: [...Object.values(incompleteCounts), ...Object.values(missingCounts)].every(
    (count) => count === 0,
  ),
  bboxFillOpacity: 0.25,
  mosaicRegion: 'detected bounding boxes',
  mosaicBlockSize: mosaicCellSize,
  mosaicBlockSizeCoordinateSpace: 'source video pixels, approximate',
  previewFrame,
  previewTimeSeconds: (previewFrame * fpsDenominator!) / fpsNumerator!,
  allFramesDecoded: true,
  modelApiCallsDuringRender: 0,
  ...(protocol.agentSettings
    ? {
        agentModels: {
          segmentation: protocol.models.sam,
          llm: protocol.agentSettings.model,
          maxLlmCalls: protocol.agentSettings.maxLlmCalls,
        },
      }
    : {}),
  renderLlmCalls: 0,
  queries: {
    sam: config.samPrompt ?? config.query,
    ...(config.samAgentMode === 'openrouter'
      ? { samAgent: 'Per-frame queries recorded in detections/sam-agent/*.json' }
      : { samAgentFixed: config.samAgentPrompt }),
    jev: config.query,
  },
  outputs,
});
console.log(JSON.stringify(outputs, null, 2));
