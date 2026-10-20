import { readFile, mkdir, readdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
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
import type { Box, Prediction } from './types.js';

// Overlay two independently acquired query results on the same frame; no model API calls.
const figureConfig = await loadConfig(
  process.argv[2] ?? 'configs/mechanical-figure-openrouter-agent.json',
);
const snsConfig = await loadConfig(
  process.argv[3] ?? 'configs/mechanical-sns-openrouter-agent.json',
);
const out = resolve(process.argv[4] ?? 'runs/blog-mechanical-comparison');
const allowMissing = process.argv.includes('--allow-missing');
const skinTarget = process.argv.includes('--skin-target');
const mosaic = process.argv.includes('--mosaic');
const mosaicCellSize = Number(
  process.argv.find((arg) => arg.startsWith('--mosaic-cell-size='))?.split('=')[1] ?? 48,
);
if (!Number.isInteger(mosaicCellSize) || mosaicCellSize < 2 || mosaicCellSize > 128)
  throw new Error('Mosaic cell size must be an integer from 2 to 128');
const targetKey = skinTarget ? 'skin' : 'sns';
const targetLabel = skinTarget ? 'exposed skin' : 'SNS-inappropriate content';
const queryLabel = skinTarget ? 'Skin' : 'SNS';
const outputStem = skinTarget ? 'figure-and-skin' : 'figure-and-sns';
const outputSuffix = mosaic ? 'bbox-mosaic' : 'bbox';
const figureProtocol = await prepare(figureConfig);
const snsProtocol = await prepare(snsConfig);
for (const key of [
  'inputSha256',
  'width',
  'height',
  'fps',
  'frameCount',
  'models',
  'thresholds',
  'agentSettings',
] as const) {
  if (JSON.stringify(figureProtocol[key]) !== JSON.stringify(snsProtocol[key]))
    throw new Error(`Query runs do not match: ${key}`);
}
const methods = methodsForConfig(figureConfig);
if (JSON.stringify(methods) !== JSON.stringify(methodsForConfig(snsConfig)))
  throw new Error('Query runs use different methods');
const labels = labelsForConfig(figureConfig).map((label, index) => {
  if (methods[index] !== 'sam-agent') return label;
  const settings = figureProtocol.agentSettings;
  if (!settings?.model) throw new Error('Agent LLM model missing from protocol');
  return `SAM Agent (SAM 3.1 + ${settings.model}, max ${settings.maxLlmCalls} LLM calls)`;
});
const files = (await readdir(join(figureConfig.runDir, 'source-frames')))
  .filter((f) => /^\d{6}\.png$/.test(f))
  .sort();
if (files.length !== figureProtocol.frameCount) throw new Error('Source frame count mismatch');
const [fpsN, fpsD] = figureProtocol.fps.split('/').map(Number);
const fps = fpsN! / fpsD!;
const previousSummary = await loadJson(join(out, 'dual-query-summary.json')).catch((error) => {
  if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
  throw error;
});
const previewFrame =
  previousSummary?.previewFrame ??
  Math.min(files.length - 1, Math.round((skinTarget ? 1.5 : 3) * fps));
const width = 1920,
  bannerHeight = 180,
  rowHeight = 600,
  height = bannerHeight + rowHeight * 2;
const blue = '#2678ff',
  red = '#ff3b45';
const frameDir = join(out, 'dual-query-frames');
await mkdir(frameDir, { recursive: true });
const xml = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const lines: string[] = [];
for (const word of snsConfig.query.split(' ')) {
  const last = lines.length - 1;
  if (last < 0 || lines[last]!.length + word.length + 1 > 132) lines.push(word);
  else lines[last] += ' ' + word;
}
if (lines.length > 2) throw new Error('SNS query exceeds banner capacity');
const banner = Buffer.from(
  `<svg width="1920" height="180"><rect width="1920" height="180" fill="#0c1421"/><g font-family="sans-serif"><text x="20" y="36" font-size="31" fill="white">${mosaic ? 'Detection outlines + mosaic' : 'Two targets in the same frame'}${allowMissing ? ' | PARTIAL RESULTS' : ''}</text><rect x="22" y="54" width="28" height="21" fill="none" stroke="${blue}" stroke-width="4"/><text x="65" y="76" font-size="26" fill="#83b4ff">Blue: figure</text><rect x="360" y="54" width="28" height="21" fill="none" stroke="${red}" stroke-width="4" stroke-dasharray="9 5"/><text x="405" y="76" font-size="26" fill="#ff858c">Red dashed: ${targetLabel}${mosaic ? ' + mosaic' : ''}</text><text x="20" y="108" font-size="21" fill="#d0d9e8">Figure query: ${xml(figureConfig.query)}</text>${lines.map((line, i) => `<text x="20" y="${136 + i * 27}" font-size="21" fill="#d0d9e8">${i === 0 ? queryLabel + ' query: ' : ''}${xml(line)}</text>`).join('')}</g></svg>`,
);
function header(title: string, figure?: Prediction, sns?: Prediction, original = false) {
  const count = (name: string, prediction?: Prediction) =>
    prediction
      ? `${name}: ${prediction.boxes.length}${prediction.status === 'incomplete' ? ' (INCOMPLETE)' : ''}`
      : `${name}: NOT PROCESSED`;
  return Buffer.from(
    `<svg width="960" height="60"><rect width="960" height="60" fill="#151b25"/><g font-family="sans-serif"><text x="16" y="25" font-size="24" fill="white">${xml(title)}</text>${!original ? `<text x="16" y="51" font-size="21" fill="#83b4ff">${xml(count('figure', figure))}</text><text x="490" y="51" font-size="21" fill="#ff858c">${xml(count(skinTarget ? 'exposed skin' : 'SNS inappropriate', sns))}</text>` : '<text x="16" y="51" font-size="20" fill="#c0c9d8">Same source frame in all panels</text>'}</g></svg>`,
  );
}
function rectangles(boxes: Box[], color: string, dashed: boolean) {
  return boxes
    .map((b) => {
      if (
        ![b.x0, b.y0, b.x1, b.y1].every(Number.isInteger) ||
        b.x0 < 0 ||
        b.y0 < 0 ||
        b.x1 > figureProtocol.width ||
        b.y1 > figureProtocol.height ||
        b.x0 >= b.x1 ||
        b.y0 >= b.y1
      )
        throw new Error('Invalid bbox geometry');
      const fill = mosaic
        ? 'fill="none"'
        : `fill="${color}" fill-opacity="${dashed ? 0.12 : 0.08}"`;
      const strokeWidth = mosaic ? (dashed ? 10 : 8) : dashed ? 6 : 5;
      return `<rect x="${b.x0}" y="${b.y0}" width="${b.x1 - b.x0}" height="${b.y1 - b.y0}" ${fill} stroke="${color}" stroke-width="${strokeWidth}"${dashed ? ' stroke-dasharray="15 10"' : ''}/>`;
    })
    .join('');
}
const counts = Object.fromEntries(
  methods.map((method) => [
    method,
    { figure: 0, sns: 0, figureIncomplete: 0, snsIncomplete: 0, figureMissing: 0, snsMissing: 0 },
  ]),
);
async function prediction(
  config: typeof figureConfig,
  method: (typeof methods)[number],
  name: string,
  frame: number,
): Promise<Prediction | undefined> {
  let result: Prediction;
  try {
    result = (await loadJson(join(config.runDir, 'detections', method, name))) as Prediction;
  } catch (error) {
    if (
      allowMissing &&
      method === 'sam-agent' &&
      (error as NodeJS.ErrnoException).code === 'ENOENT'
    )
      return undefined;
    throw error;
  }
  assertPrediction(config, method, result, frame);
  if (Math.abs(result.timeSeconds - frame / fps) > 1e-8)
    throw new Error('Query result timestamps differ');
  return result;
}
for (let i = 0; i < files.length; i++) {
  const file = files[i]!;
  const source = await readFile(join(figureConfig.runDir, 'source-frames', file));
  const tiles: { input: Buffer; left: number; top: number }[] = [
    { input: banner, left: 0, top: 0 },
    {
      input: header(
        `Original | ${(i / fps).toFixed(2)} s | frame ${i + 1}/${files.length}`,
        undefined,
        undefined,
        true,
      ),
      left: 0,
      top: bannerHeight,
    },
    {
      input: await sharp(source).resize(960, 540).png().toBuffer(),
      left: 0,
      top: bannerHeight + 60,
    },
  ];
  for (let m = 0; m < methods.length; m++) {
    const method = methods[m]!;
    const name = file.replace('.png', '.json');
    const figure = await prediction(figureConfig, method, name, i);
    const sns = await prediction(snsConfig, method, name, i);
    const c = counts[method]!;
    if (figure?.boxes.length) c.figure++;
    if (sns?.boxes.length) c.sns++;
    if (figure?.status === 'incomplete') c.figureIncomplete++;
    if (sns?.status === 'incomplete') c.snsIncomplete++;
    if (!figure) c.figureMissing++;
    if (!sns) c.snsMissing++;
    const overlay = Buffer.from(
      `<svg width="${figureProtocol.width}" height="${figureProtocol.height}">${rectangles(figure?.boxes ?? [], blue, false)}${rectangles(sns?.boxes ?? [], red, true)}</svg>`,
    );
    const base = mosaic
      ? (await panels(source, sns?.boxes ?? [], true, mosaicCellSize)).mosaic
      : source;
    const overlaid = await sharp(base)
      .composite([{ input: overlay }])
      .png()
      .toBuffer();
    const image = await sharp(overlaid).resize(960, 540).png().toBuffer();
    const index = m + 1,
      left = (index % 2) * 960,
      top = bannerHeight + Math.floor(index / 2) * rowHeight;
    tiles.push(
      { input: header(labels[m]!, figure, sns), left, top },
      { input: image, left, top: top + 60 },
    );
  }
  const frame = await sharp({ create: { width, height, channels: 3, background: '#151b25' } })
    .composite(tiles)
    .png()
    .toBuffer();
  await writeFile(join(frameDir, file), frame);
  if (i === previewFrame) await writeFile(join(out, `${outputStem}-preview.png`), frame);
  if ((i + 1) % 20 === 0 || i === files.length - 1)
    console.log(`Dual-query render: ${i + 1}/${files.length}`);
}
const output = join(out, `${outputStem}-${outputSuffix}.mp4`);
execFileSync('ffmpeg', [
  '-hide_banner',
  '-loglevel',
  'error',
  '-y',
  '-framerate',
  figureProtocol.fps,
  '-i',
  join(frameDir, '%06d.png'),
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
  stream.avg_frame_rate !== figureProtocol.fps ||
  stream.width !== width ||
  stream.height !== height
)
  throw new Error('Rendered video metadata mismatch');
execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', output, '-f', 'null', '-']);
const summaryCounts = skinTarget
  ? Object.fromEntries(
      Object.entries(counts).map(([method, c]) => [
        method,
        {
          figure: c.figure,
          skin: c.sns,
          figureIncomplete: c.figureIncomplete,
          skinIncomplete: c.snsIncomplete,
          figureMissing: c.figureMissing,
          skinMissing: c.snsMissing,
        },
      ]),
    )
  : counts;
await saveJson(join(out, 'dual-query-summary.json'), {
  output,
  video,
  inputSha256: figureProtocol.inputSha256,
  frames: files.length,
  queries: { figure: figureConfig.query, [targetKey]: snsConfig.query },
  colors: { figure: { color: blue, line: 'solid' }, [targetKey]: { color: red, line: 'dashed' } },
  redTarget: targetLabel,
  ...(mosaic
    ? {
        visualization: 'Outline-only bounding boxes with mosaic inside red target boxes',
        bboxFillOpacity: 0,
        strokeWidthsSourcePixels: { figure: 8, [targetKey]: 10 },
        mosaicRegion: 'red target bounding boxes only; blue figure boxes are outlines only',
        mosaicBlockSize: mosaicCellSize,
        mosaicBlockSizeCoordinateSpace: 'source video pixels, approximate',
      }
    : {}),
  ...(figureProtocol.agentSettings
    ? {
        agentModels: {
          segmentation: figureProtocol.models.sam,
          llm: figureProtocol.agentSettings.model,
          maxLlmCalls: figureProtocol.agentSettings.maxLlmCalls,
        },
      }
    : {}),
  counts: summaryCounts,
  partial: Object.values(counts).some((c) => c.figureMissing > 0 || c.snsMissing > 0),
  layout: [
    ['Original', labels[0]],
    [labels[1], labels[2]],
  ],
  figureRunDir: figureConfig.runDir,
  [`${targetKey}RunDir`]: snsConfig.runDir,
  inferenceMode:
    'Two independent cached query runs over the same frames, overlaid in one video; not a new joint-query inference.',
  modelApiCallsDuringRender: 0,
  allFramesDecoded: true,
  previewFrame,
  previewTimeSeconds: previewFrame / fps,
});
console.log(JSON.stringify({ output, counts: summaryCounts, video }, null, 2));
