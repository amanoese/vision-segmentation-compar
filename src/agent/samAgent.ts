import { readFile, mkdir, writeFile, access } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import sharp from 'sharp';
import { z } from 'zod';
import { runSamBridge } from '../sam/agentBridge.js';
import { renderOverlay } from '../sam/render.js';
import { maskPixels } from '../sam/response.js';
import { pixelBoxes } from '../render.js';
import type { Box } from '../types.js';
import {
  OpenRouterClient,
  requestWithoutImages,
  type AgentSettings,
  type LlmClient,
  type LlmRequest,
  type Message,
  type LlmReply,
} from './openrouter.js';

const exists = (path: string) =>
  access(path).then(
    () => true,
    () => false,
  );
const json = async (path: string) => JSON.parse(await readFile(path, 'utf8'));
const save = async (path: string, data: unknown) =>
  writeFile(path, JSON.stringify(data, null, 2) + '\n');
import { SYSTEM_PROMPT } from './prompt.js';

const reason = z.string().trim().min(1).max(400);
const segmentSchema = z.object({ text_prompt: z.string().trim().min(1).max(180), reason }).strict();
const selectionSchema = z
  .object({ mask_ids: z.array(z.number().int().positive()).min(1), reason })
  .strict();
const absentSchema = z.object({ reason }).strict();
function tools(final: boolean): LlmRequest['tools'] {
  const tool = (name: string, description: string, properties: object, required: string[]) => ({
    type: 'function' as const,
    function: {
      name,
      description,
      parameters: { type: 'object', properties, required, additionalProperties: false },
    },
  });
  const reasonProperty = {
    type: 'string',
    description: 'One short sentence based on visible evidence.',
  };
  return [
    ...(!final
      ? [
          tool(
            'segment_phrase',
            'Detect regions with a concrete English noun phrase using SAM 3.1.',
            {
              text_prompt: { type: 'string', maxLength: 180 },
              reason: reasonProperty,
            },
            ['text_prompt', 'reason'],
          ),
        ]
      : []),
    tool(
      'select_masks_and_return',
      'Accept matching masks from the CURRENT numbered overlay and finish.',
      {
        mask_ids: { type: 'array', items: { type: 'integer', minimum: 1 }, minItems: 1 },
        reason: reasonProperty,
      },
      ['mask_ids', 'reason'],
    ),
    tool(
      'report_no_mask',
      'Finish only if the image has no visible regions matching the ORIGINAL request.',
      {
        reason: reasonProperty,
      },
      ['reason'],
    ),
  ];
}
export interface Candidates {
  boxes: Box[];
  scores: number[];
  overlayPath: string;
  maskPaths: string[];
  recoveredFromSavedResponse?: boolean;
}
export interface AgentResult {
  status: 'detected' | 'no-detections' | 'incomplete';
  boxes: Box[];
  scores: number[];
  queryHistory: string[];
  settings: AgentSettings;
  llmCalls: number;
  samCalls: number;
  elapsedMs: number;
  timingSource: 'fresh' | 'resumed';
  boxSource: 'llm-selected' | 'unreviewed-candidates' | 'none';
  finishReason: string;
  reason?: string;
}
export interface AgentOptions {
  image: string;
  out: string;
  query: string;
  settings: AgentSettings;
}
export interface AgentDependencies {
  llm?: LlmClient;
  segment?: (query: string, out: string) => Promise<Candidates>;
}
async function imagePart(
  path: string,
): Promise<Extract<Exclude<Message['content'], string>[number], { type: 'image_url' }>> {
  const jpeg = await sharp(await readFile(path))
    .rotate()
    .resize({ width: 1280, height: 1280, fit: 'inside', withoutEnlargement: true })
    .flatten({ background: 'white' })
    .jpeg({ quality: 85 })
    .toBuffer();
  return {
    type: 'image_url',
    image_url: { url: `data:image/jpeg;base64,${jpeg.toString('base64')}` },
  };
}
async function segmentWithSam(image: string, out: string, query: string): Promise<Candidates> {
  if (await exists(join(out, 'candidates.json'))) return json(join(out, 'candidates.json'));
  const result = await runSamBridge({ image, out, prompt: query, resumeSaved: await exists(out) });
  const boxes = pixelBoxes(result.pred_boxes, result.orig_img_w, result.orig_img_h);
  if (boxes.length !== result.pred_scores.length || boxes.length !== result.png_masks.length)
    throw new Error('agent-sam-candidate-count-mismatch');
  // IDs are positional and identical in the overlay, table and returned boxes.
  const foreground = await Promise.all(
    result.png_masks.map(
      async (path) =>
        (await maskPixels(await readFile(path), result.orig_img_w, result.orig_img_h)).foreground,
    ),
  );
  const overlayPath = join(out, 'agent-numbered-masks.png');
  await writeFile(
    overlayPath,
    await renderOverlay(
      await readFile(image),
      boxes.map((box, index) => ({
        index,
        box,
        score: result.pred_scores[index]!,
        boxSource: 'api' as const,
      })),
      foreground,
      result.orig_img_w,
      result.orig_img_h,
    ),
  );
  const candidates = {
    boxes,
    scores: result.pred_scores,
    overlayPath,
    maskPaths: result.png_masks,
    recoveredFromSavedResponse: result.recoveredFromSavedResponse,
  };
  await save(join(out, 'candidates.json'), candidates);
  return candidates;
}

export async function runSamAgent(
  options: AgentOptions,
  dependencies: AgentDependencies = {},
): Promise<AgentResult> {
  const started = performance.now();
  if (options.settings.maxLlmCalls !== 3) throw new Error('agent-budget-must-be-three');
  const imageSha256 = createHash('sha256')
    .update(await readFile(options.image))
    .digest('hex');
  const manifest = {
    image: resolve(options.image),
    imageSha256,
    query: options.query,
    settings: options.settings,
    systemPrompt: SYSTEM_PROMPT,
  };
  await mkdir(options.out, { recursive: true });
  const manifestPath = join(options.out, 'request.json');
  if (await exists(manifestPath)) {
    if (JSON.stringify(await json(manifestPath)) !== JSON.stringify(manifest))
      throw new Error('agent-saved-conditions-mismatch');
  } else await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  const resultPath = join(options.out, 'result.json');
  if (await exists(resultPath)) return json(resultPath);
  // A directory lock prevents simultaneous invocations from duplicating paid calls.
  const lock = join(options.out, '.active');
  try {
    await mkdir(lock);
  } catch {
    throw new Error('agent-run-already-active-or-interrupted');
  }
  const { rmdir } = await import('node:fs/promises');
  let llm: LlmClient | undefined = dependencies.llm;
  const segment =
    dependencies.segment ?? ((query, out) => segmentWithSam(options.image, out, query));
  let candidates: Candidates | undefined;
  let resumed = false;
  let feedback = 'No SAM candidates have been generated.';
  let llmCalls = 0,
    samCalls = 0;
  const queryHistory: string[] = [];
  async function finish(
    status: AgentResult['status'],
    finishReason: string,
    ids: number[] = [],
    why?: string,
  ) {
    // Keep the latest SAM result on an unfinished decision. Do not resurrect
    // candidates from an older query if the latest query produced no boxes.
    const fallback = status === 'incomplete' && Boolean(candidates?.boxes.length);
    const result: AgentResult = {
      status,
      finishReason,
      settings: options.settings,
      llmCalls,
      samCalls,
      queryHistory,
      elapsedMs: Math.round(performance.now() - started),
      timingSource: resumed ? 'resumed' : 'fresh',
      boxSource: fallback ? 'unreviewed-candidates' : ids.length ? 'llm-selected' : 'none',
      boxes: fallback ? candidates!.boxes : ids.map((id) => candidates!.boxes[id - 1]!),
      scores: fallback ? candidates!.scores : ids.map((id) => candidates!.scores[id - 1]!),
      ...(why ? { reason: why } : {}),
    };
    await save(resultPath, result);
    return result;
  }
  try {
    const originalImage = await imagePart(options.image);
    for (let turn = 1; turn <= options.settings.maxLlmCalls; turn++) {
      const stem = join(options.out, `turn-${String(turn).padStart(2, '0')}`);
      const final = turn === options.settings.maxLlmCalls;
      const content: Exclude<Message['content'], string> = [
        { type: 'text', text: `Original request: ${options.query}\nCurrent raw input image:` },
        originalImage,
        {
          type: 'text',
          text: `LLM call ${turn}/${options.settings.maxLlmCalls}. ${final ? 'FINAL call: do not request another segmentation.' : 'You may segment or finish.'}\nPreviously tried phrases: ${JSON.stringify(queryHistory)}\nFeedback: ${feedback}`,
        },
      ];
      if (candidates) {
        content.push({
          type: 'text',
          text: `Current candidates (1-based IDs): ${JSON.stringify(candidates.boxes.map((box, i) => ({ id: i + 1, box, score: candidates!.scores[i] })))}\nCurrent numbered masks:`,
        });
        content.push(await imagePart(candidates.overlayPath));
      }
      const request: LlmRequest = {
        model: options.settings.model,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content },
        ],
        tools: tools(final),
        tool_choice: 'required',
        parallel_tool_calls: false,
        max_tokens: options.settings.maxOutputTokens,
        reasoning: { enabled: options.settings.reasoningEnabled },
      };
      let reply: LlmReply;
      if (await exists(stem + '.response.json')) {
        resumed = true;
        reply = await json(stem + '.response.json');
      } else {
        if (await exists(stem + '.request.json'))
          throw new Error('agent-llm-outcome-unknown-do-not-resubmit');
        llm ??= new OpenRouterClient(process.env.OPENROUTER_API_KEY ?? '');
        await writeFile(
          stem + '.request.json',
          JSON.stringify(requestWithoutImages(request), null, 2) + '\n',
          { flag: 'wx' },
        );
        // Each attempted POST consumes one of the three slots, including malformed/truncated replies.
        try {
          reply = await llm.complete(request);
          await save(stem + '.response.json', reply);
        } catch (error) {
          // Only transport-generated safe codes are written; never arbitrary provider payloads.
          const code =
            error instanceof Error && /^openrouter-[a-z0-9-]+$/.test(error.message)
              ? error.message
              : 'agent-llm-failed';
          await save(stem + '.error.json', { code, outcomeUnknown: true });
          throw new Error(code);
        }
      }
      llmCalls++;
      const choice = reply.choices[0];
      if (!choice || choice.finish_reason === 'length' || choice.finish_reason === 'content_filter')
        return await finish(
          'incomplete',
          choice?.finish_reason === 'length' ? 'output-limit' : 'llm-no-usable-output',
        );
      const calls = choice.message.tool_calls;
      if (calls?.length !== 1) {
        feedback = 'Invalid response: return exactly one supported tool call.';
        continue;
      }
      const call = calls[0]!;
      let args: unknown;
      try {
        args = JSON.parse(call.function.arguments);
      } catch {
        feedback = 'Invalid tool arguments: use valid JSON.';
        continue;
      }
      if (call.function.name === 'segment_phrase') {
        const parsed = segmentSchema.safeParse(args);
        if (
          final ||
          !parsed.success ||
          queryHistory.includes(parsed.success ? parsed.data.text_prompt : '')
        ) {
          feedback = 'Segmentation rejected: final call, invalid arguments, or repeated phrase.';
          continue;
        }
        const prompt = parsed.data.text_prompt;
        queryHistory.push(prompt);
        const samOut = stem + '.sam';
        samCalls++;
        if (await exists(stem + '.candidates.json')) {
          resumed = true;
          candidates = await json(stem + '.candidates.json');
        } else candidates = await segment(prompt, samOut);
        if (candidates?.recoveredFromSavedResponse) resumed = true;
        await save(stem + '.candidates.json', candidates);
        feedback = `SAM returned ${candidates!.boxes.length} candidates for ${JSON.stringify(prompt)}. Inspect visible regions against the original request. Zero candidates alone does not establish absence.`;
      } else if (call.function.name === 'select_masks_and_return') {
        const parsed = selectionSchema.safeParse(args);
        if (
          !parsed.success ||
          !candidates ||
          new Set(parsed.data.mask_ids).size !== parsed.data.mask_ids.length ||
          parsed.data.mask_ids.some((id) => id > candidates!.boxes.length)
        ) {
          feedback = 'Invalid selection: use unique IDs from the CURRENT candidates only.';
          continue;
        }
        return await finish('detected', 'selected-masks', parsed.data.mask_ids, parsed.data.reason);
      } else if (call.function.name === 'report_no_mask') {
        const parsed = absentSchema.safeParse(args);
        if (parsed.success)
          return await finish('no-detections', 'llm-confirmed-absence', [], parsed.data.reason);
        feedback = 'Invalid absence report: provide a short visible-evidence reason.';
      } else feedback = 'Unsupported tool. Use one of the supplied tools.';
    }
    return await finish('incomplete', 'llm-call-limit');
  } finally {
    await rmdir(lock);
  }
}
