import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import sharp from 'sharp';
import { ENDPOINT, SamTransport, type Inference } from './transport.js';
import { parseOutput, maskPixels, normalizedBox } from './response.js';
import { renderOverlay, type Detection } from './render.js';

export async function runSamBridge(
  options: { image: string; out: string; prompt: string; resumeSaved?: boolean },
  injected?: Inference,
) {
  const credentials = process.env.FALAI_API_KEY ?? process.env.FAL_KEY;
  if (!credentials && !injected && !options.resumeSaved) throw new Error('fal-key-missing');
  const source = await sharp(await readFile(options.image))
    .rotate()
    .png()
    .toBuffer();
  const { width, height } = await sharp(source).metadata();
  if (!width || !height) throw new Error('image-dimensions-missing');
  const settings = {
    prompt: options.prompt,
    apply_mask: false,
    output_format: 'png',
    return_multiple_masks: true,
    max_masks: 10,
    include_scores: true,
    include_boxes: true,
  };
  let result: { data: unknown; requestId: string; elapsedMs: number };
  if (options.resumeSaved) {
    const previous = JSON.parse(await readFile(join(options.out, 'request.json'), 'utf8'));
    if (
      previous.image !== resolve(options.image) ||
      previous.prompt !== options.prompt ||
      previous.endpoint !== ENDPOINT
    )
      throw new Error('saved-sam-request-mismatch');
    const job = JSON.parse(await readFile(join(options.out, 'job.json'), 'utf8'));
    result = {
      data: JSON.parse(await readFile(join(options.out, 'response.json'), 'utf8')),
      requestId: job.requestId,
      elapsedMs: 0,
    };
  } else {
    await mkdir(options.out, { recursive: false });
    await writeFile(
      join(options.out, 'request.json'),
      JSON.stringify({ endpoint: ENDPOINT, ...settings, image: resolve(options.image) }, null, 2),
    );
    const transport = injected ?? new SamTransport(credentials!);
    result = await transport.infer(
      { image_url: `data:image/png;base64,${source.toString('base64')}`, ...settings },
      async (requestId) => {
        await writeFile(
          join(options.out, 'job.json'),
          JSON.stringify({ requestId, endpoint: ENDPOINT }, null, 2),
        );
      },
    );
    await writeFile(join(options.out, 'response.json'), JSON.stringify(result.data, null, 2));
  }
  const parsed = parseOutput(result.data);
  const detections: Detection[] = [],
    foreground: Buffer[] = [],
    png_masks: string[] = [],
    pred_boxes: number[][] = [],
    pred_scores: number[] = [];
  for (const item of parsed.masks) {
    let png: Buffer;
    if (item.url.startsWith('data:image/png;base64,'))
      png = Buffer.from(item.url.split(',')[1]!, 'base64');
    else {
      if (new URL(item.url).protocol !== 'https:') throw new Error('invalid-mask-url');
      const response = await fetch(item.url, { signal: AbortSignal.timeout(30_000) });
      if (!response.ok) throw new Error('mask-download-failed');
      png = Buffer.from(await response.arrayBuffer());
    }
    const mask = await maskPixels(png, width, height);
    if (!mask.box || item.score === undefined) throw new Error('mask-or-score-missing');
    const box = item.normalized ? normalizedBox(item.normalized, width, height) : mask.box;
    const pngPath = resolve(options.out, `mask-${item.index}.png`);
    await writeFile(pngPath, png);
    png_masks.push(pngPath);
    foreground.push(mask.foreground);
    detections.push({
      index: item.index,
      box,
      score: item.score,
      boxSource: item.normalized ? 'api' : 'mask-derived',
      foregroundPixels: mask.foregroundPixels,
    });
    if (item.normalized) {
      const [cx, cy, w, h] = item.normalized;
      pred_boxes.push([cx - w / 2, cy - h / 2, w, h]);
    } else
      pred_boxes.push([
        box.x0 / width,
        box.y0 / height,
        (box.x1 - box.x0) / width,
        (box.y1 - box.y0) / height,
      ]);
    pred_scores.push(item.score);
  }
  const output_image_path = resolve(options.out, 'overlay.png');
  await writeFile(
    output_image_path,
    await renderOverlay(source, detections, foreground, width, height),
  );
  const output = {
    original_image_path: resolve(options.image),
    orig_img_w: width,
    orig_img_h: height,
    pred_boxes,
    pred_scores,
    png_masks,
    output_image_path,
    requestId: result.requestId,
    elapsedMs: result.elapsedMs,
    prompt: options.prompt,
    recoveredFromSavedResponse: Boolean(options.resumeSaved),
  };
  await writeFile(join(options.out, 'bridge.json'), JSON.stringify(output, null, 2));
  return output;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { values } = parseArgs({
    options: { image: { type: 'string' }, out: { type: 'string' }, prompt: { type: 'string' } },
  });
  if (!values.image || !values.out || !values.prompt) throw new Error('bridge-arguments-missing');
  runSamBridge({ image: values.image, out: values.out, prompt: values.prompt })
    .then((result) => console.log(JSON.stringify(result)))
    .catch(() => {
      console.error('sam-bridge-failed: see saved job/response');
      process.exitCode = 1;
    });
}
