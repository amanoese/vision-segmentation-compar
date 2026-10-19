import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join, dirname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import sharp from 'sharp';
import { z } from 'zod';

const probability = z.number().min(0).max(1);
const boxSchema = z.object({
  top: probability,
  bottom: probability,
  left: probability,
  right: probability,
});
const responseSchema = z.object({
  model: z.string(),
  answers: z.object({
    present: z.object({ type: z.literal('noul'), noul: probability }),
    location: z.object({ type: z.literal('box'), box: boxSchema, confidence: probability }),
  }),
});
export const DEFAULT_TARGET = '肌露出';
export const ENDPOINT = 'https://api.typesafe.ai/v1/systemone';
export const MODEL = 'jev-1.14.0';

// Keep the aspect ratio, then pad narrow images to the API minimum.
// Coordinates in result.json refer to the upright original, before resize/padding.
export async function prepareImage(input: Buffer) {
  const metadata = await sharp(input).metadata();
  if (!['png', 'jpeg'].includes(metadata.format ?? '') || (metadata.pages ?? 1) !== 1)
    throw new Error('Input must be a still PNG or JPEG');
  const original = await sharp(input).rotate().flatten({ background: '#ffffff' }).png().toBuffer();
  const size = await sharp(original).metadata();
  const width = size.width!,
    height = size.height!;
  const scale = Math.min(1, 1568 / Math.max(width, height));
  const resizedWidth = Math.max(1, Math.round(width * scale));
  const resizedHeight = Math.max(1, Math.round(height * scale));
  const preparedWidth = Math.max(280, resizedWidth),
    preparedHeight = Math.max(280, resizedHeight);
  const left = Math.floor((preparedWidth - resizedWidth) / 2);
  const top = Math.floor((preparedHeight - resizedHeight) / 2);
  const jpeg = await sharp(original)
    .resize(resizedWidth, resizedHeight)
    .extend({
      left,
      top,
      right: preparedWidth - resizedWidth - left,
      bottom: preparedHeight - resizedHeight - top,
      background: '#ffffff',
    })
    .jpeg({ quality: 90, progressive: false })
    .toBuffer();
  if (jpeg.length > 16 * 1024 * 1024) throw new Error('Prepared image exceeds 16 MiB');
  return {
    original,
    jpeg,
    width,
    height,
    resizedWidth,
    resizedHeight,
    preparedWidth,
    preparedHeight,
    left,
    top,
  };
}
type Prepared = Awaited<ReturnType<typeof prepareImage>>;

export function buildJevRequest(image: Prepared, target: string) {
  const reference = randomUUID();
  return {
    model: MODEL,
    state: { photo: reference, target },
    assets: {
      [reference]: { type: 'image', format: 'jpg', data_base64: image.jpeg.toString('base64') },
    },
    questions: {
      present: {
        type: 'noul',
        instructions: 'Is at least one item matching `target` visible in `photo`?',
      },
      location: {
        type: 'box',
        instructions:
          'The single most prominent item matching `target` in `photo`. Enclose the whole item tightly. If several match, locate only the most prominent one.',
      },
    },
  };
}

export function interpretResponse(
  raw: unknown,
  image: Prepared,
  presenceThreshold: number,
  boxThreshold: number,
) {
  const response = responseSchema.parse(raw);
  if (response.model !== MODEL) throw new Error(`Unexpected response model: ${response.model}`);
  const { present, location } = response.answers;
  const b = location.box;
  const clamp = (v: number, max: number) => Math.min(max, Math.max(0, v));
  const bbox = {
    x0: clamp(
      Math.floor(((b.left * image.preparedWidth - image.left) * image.width) / image.resizedWidth),
      image.width,
    ),
    y0: clamp(
      Math.floor(((b.top * image.preparedHeight - image.top) * image.height) / image.resizedHeight),
      image.height,
    ),
    x1: clamp(
      Math.ceil(((b.right * image.preparedWidth - image.left) * image.width) / image.resizedWidth),
      image.width,
    ),
    y1: clamp(
      Math.ceil(
        ((b.bottom * image.preparedHeight - image.top) * image.height) / image.resizedHeight,
      ),
      image.height,
    ),
  };
  const geometryValid =
    b.left < b.right && b.top < b.bottom && bbox.x0 < bbox.x1 && bbox.y0 < bbox.y1;
  const status =
    present.noul <= 1 - presenceThreshold
      ? 'absent'
      : present.noul < presenceThreshold
        ? 'uncertain-presence'
        : !geometryValid
          ? 'invalid-box'
          : location.confidence < boxThreshold
            ? 'uncertain-box'
            : 'detected';
  return {
    status,
    presenceProbability: present.noul,
    boxConfidence: location.confidence,
    normalizedBox: b,
    bbox: status === 'detected' ? bbox : null,
  };
}

export interface Options {
  image: string;
  runDir: string;
  target?: string;
  dryRun?: boolean;
  presenceThreshold?: number;
  boxThreshold?: number;
}
export async function runJevImage(
  options: Options,
  deps: { apiKey?: string; fetcher?: typeof fetch } = {},
) {
  const target = (options.target ?? DEFAULT_TARGET).trim();
  if (!target || target.length > 2000) throw new Error('Target must contain 1–2000 characters');
  const presenceThreshold = z
    .number()
    .gt(0.5)
    .max(1)
    .parse(options.presenceThreshold ?? 0.8);
  const boxThreshold = probability.parse(options.boxThreshold ?? 0.6);
  const key = deps.apiKey ?? (process.env.TYPESAFE_API_KEY || process.env.JEV_API_KEY);
  if (!options.dryRun && !key) throw new Error('Set TYPESAFE_API_KEY (or JEV_API_KEY)');
  const input = await readFile(options.image);
  const prepared = await prepareImage(input);
  const request = buildJevRequest(prepared, target);
  const body = JSON.stringify(request);
  if (Buffer.byteLength(body) > 32 * 1024 * 1024) throw new Error('Request exceeds 32 MiB');
  // Atomic directory creation prevents accidentally overwriting earlier evidence.
  await mkdir(dirname(resolve(options.runDir)), { recursive: true });
  await mkdir(resolve(options.runDir), { recursive: false });
  const save = (name: string, value: unknown) =>
    writeFile(join(options.runDir, name), JSON.stringify(value, null, 2) + '\n');
  const protocol = {
    endpoint: ENDPOINT,
    model: MODEL,
    target,
    dryRun: Boolean(options.dryRun),
    input: resolve(options.image),
    inputSha256: createHash('sha256').update(input).digest('hex'),
    original: { width: prepared.width, height: prepared.height, orientation: 'EXIF applied' },
    prepared: {
      width: prepared.preparedWidth,
      height: prepared.preparedHeight,
      resizedWidth: prepared.resizedWidth,
      resizedHeight: prepared.resizedHeight,
      left: prepared.left,
      top: prepared.top,
    },
    thresholds: { presence: presenceThreshold, box: boxThreshold },
    request: {
      ...request,
      assets: Object.fromEntries(
        Object.keys(request.assets).map((k) => [
          k,
          { type: 'image', format: 'jpg', file: 'prepared.jpg' },
        ]),
      ),
    },
    coordinateSystem: 'upright original pixels; x1/y1 exclusive',
  };
  await save('protocol.json', protocol);
  await writeFile(join(options.runDir, 'prepared.jpg'), prepared.jpeg);
  await writeFile(join(options.runDir, 'original.png'), prepared.original);
  if (options.dryRun) return { status: 'dry-run' };
  const started = performance.now();
  try {
    // One image, one request; no automatic retries or fallback models.
    const response = await (deps.fetcher ?? fetch)(ENDPOINT, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${key}`,
        'Content-Type': 'application/json',
        'TypeSafe-Danger-Zone-Images': 'true',
      },
      body,
      signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) {
      await response.body?.cancel();
      throw new Error(`JEV HTTP ${response.status}`);
    }
    const raw: unknown = await response.json();
    await save('response.json', raw);
    const result = {
      ...interpretResponse(raw, prepared, presenceThreshold, boxThreshold),
      httpMs: performance.now() - started,
    };
    if (result.bbox) {
      const b = result.bbox;
      const overlay = Buffer.from(
        `<svg width="${prepared.width}" height="${prepared.height}"><rect x="${b.x0}" y="${b.y0}" width="${b.x1 - b.x0}" height="${b.y1 - b.y0}" fill="#0066ff" stroke="#0066ff" stroke-width="4" fill-opacity="0.25"/></svg>`,
      );
      await sharp(prepared.original)
        .composite([{ input: overlay }])
        .png()
        .toFile(join(options.runDir, 'annotated.png'));
      await sharp(prepared.original)
        .extract({ left: b.x0, top: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0 })
        .png()
        .toFile(join(options.runDir, 'crop.png'));
    }
    await save('result.json', result);
    return result;
  } catch (error) {
    await save('error.json', {
      status: 'failed',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
    throw error;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { values } = parseArgs({
    options: {
      image: { type: 'string' },
      'run-dir': { type: 'string' },
      target: { type: 'string' },
      'dry-run': { type: 'boolean', default: false },
      'presence-threshold': { type: 'string', default: '0.8' },
      'box-threshold': { type: 'string', default: '0.6' },
    },
  });
  if (!values.image || !values['run-dir']) {
    console.error(
      'Usage: pnpm jev:image --image photo.png --run-dir runs/jev-test [--target "a plush toy"] [--dry-run]',
    );
    process.exitCode = 1;
  } else {
    runJevImage({
      image: values.image,
      runDir: values['run-dir'],
      ...(values.target ? { target: values.target } : {}),
      dryRun: values['dry-run'],
      presenceThreshold: Number(values['presence-threshold']),
      boxThreshold: Number(values['box-threshold']),
    })
      .then((result) => console.log(JSON.stringify(result, null, 2)))
      .catch((error) => {
        console.error(error instanceof Error ? error.message : 'JEV image PoC failed');
        process.exitCode = 1;
      });
  }
}
