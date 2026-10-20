import { mkdtemp, readFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import sharp from 'sharp';
import { afterEach, expect, it, vi } from 'vitest';
import { runSamBridge } from '../src/sam/agentBridge.js';
import { runSamAgent, type Candidates } from '../src/agent/samAgent.js';
import {
  agentSettingsFromEnv,
  OpenRouterClient,
  type LlmReply,
  type LlmRequest,
} from '../src/agent/openrouter.js';

vi.mock('../src/sam/agentBridge.js', () => ({ runSamBridge: vi.fn() }));

const roots: string[] = [];
afterEach(async () => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  await Promise.all(roots.splice(0).map((p) => rm(p, { recursive: true, force: true })));
});
async function setup() {
  const root = await mkdtemp(join(tmpdir(), 'sam-openrouter-test-'));
  roots.push(root);
  const image = join(root, 'source.png');
  await sharp({ create: { width: 280, height: 280, channels: 3, background: 'gray' } })
    .png()
    .toFile(image);
  const first: Candidates = {
    boxes: [{ x0: 20, y0: 30, x1: 90, y1: 100 }],
    scores: [0.9],
    overlayPath: image,
    maskPaths: [],
  };
  const second: Candidates = {
    boxes: [{ x0: 110, y0: 120, x1: 200, y1: 240 }],
    scores: [0.8],
    overlayPath: image,
    maskPaths: [],
  };
  vi.stubEnv('SAM_AGENT_MODEL', 'qwen/qwen3.7-flash');
  return {
    root,
    first,
    second,
    options: {
      image,
      out: join(root, 'agent'),
      query: 'Excessive skin exposure',
      settings: agentSettingsFromEnv(),
    },
  };
}
function reply(name: string, args: object, finishReason = 'tool_calls'): LlmReply {
  return {
    model: 'qwen/qwen3.7-flash',
    id: 'mock-response',
    usage: { prompt_tokens: 100, completion_tokens: 25, cost: 0 },
    choices: [
      {
        finish_reason: finishReason,
        message: {
          tool_calls: [
            { id: 'tool-1', type: 'function', function: { name, arguments: JSON.stringify(args) } },
          ],
        },
      },
    ],
  };
}

it('uses three LLM calls for initial query, revision and reviewed selection, and caches without extra API calls', async () => {
  const { options, first, second } = await setup();
  const requests: LlmRequest[] = [];
  const responses = [
    reply('segment_phrase', { text_prompt: 'bare torso', reason: 'Locate the exposed torso.' }),
    reply('segment_phrase', {
      text_prompt: 'exposed skin on chest',
      reason: 'The first candidate covered armor.',
    }),
    reply('select_masks_and_return', {
      mask_ids: [1],
      reason: 'The current candidate contains visible bare skin.',
    }),
  ];
  const complete = vi.fn(async (r: LlmRequest) => {
    requests.push(r);
    return responses[requests.length - 1]!;
  });
  const segment = vi.fn(async () => (segment.mock.calls.length === 1 ? first : second));
  const result = await runSamAgent(options, { llm: { complete }, segment });
  expect(result).toMatchObject({
    status: 'detected',
    boxes: second.boxes,
    scores: second.scores,
    llmCalls: 3,
    samCalls: 2,
    queryHistory: ['bare torso', 'exposed skin on chest'],
  });
  expect(complete).toHaveBeenCalledTimes(3);
  expect(segment).toHaveBeenCalledTimes(2);
  expect(
    requests.every(
      (r) =>
        r.max_tokens === 1024 && r.model === 'qwen/qwen3.7-flash' && r.reasoning.enabled === false,
    ),
  ).toBe(true);
  expect(requests[2]!.tools.map((t) => t.function.name)).not.toContain('segment_phrase');
  const secondContent = requests[1]!.messages[1]!.content;
  expect(
    Array.isArray(secondContent) && secondContent.filter((c) => c.type === 'image_url').length,
  ).toBe(2);
  const savedRequest = await readFile(join(options.out, 'turn-01.request.json'), 'utf8');
  expect(savedRequest).not.toContain('base64');
  expect(savedRequest).not.toContain('Authorization');
  expect(await runSamAgent(options, { llm: { complete }, segment })).toEqual(result);
  expect(complete).toHaveBeenCalledTimes(3);
  await rm(join(options.out, 'result.json'));
  const replayed = await runSamAgent(options, { llm: { complete }, segment });
  expect(replayed.boxes).toEqual(second.boxes);
  expect(replayed.timingSource).toBe('resumed');
  expect(complete).toHaveBeenCalledTimes(3);
  expect(segment).toHaveBeenCalledTimes(2);
});

it('never launches a fourth LLM call or third SAM request even if the final response requests segmentation', async () => {
  const { options, first, second } = await setup();
  let clock = 100;
  vi.spyOn(performance, 'now').mockImplementation(() => clock);
  let turn = 0;
  const complete = vi.fn(
    async () => (
      (clock += 100),
      reply('segment_phrase', { text_prompt: `candidate ${++turn}`, reason: 'Try a target.' })
    ),
  );
  const segment = vi.fn(async () => {
    clock += 60;
    return segment.mock.calls.length === 1 ? first : second;
  });
  const result = await runSamAgent(options, { llm: { complete }, segment });
  expect(result).toMatchObject({
    status: 'incomplete',
    finishReason: 'llm-call-limit',
    boxSource: 'unreviewed-candidates',
    elapsedMs: 420,
    timingSource: 'fresh',
    boxes: second.boxes,
    scores: second.scores,
    llmCalls: 3,
    samCalls: 2,
  });
  expect(complete).toHaveBeenCalledTimes(3);
  expect(segment).toHaveBeenCalledTimes(2);
});

it('allows absence directly from the raw image without invoking SAM', async () => {
  const { options } = await setup();
  const segment = vi.fn();
  const result = await runSamAgent(options, {
    llm: {
      complete: async () =>
        reply('report_no_mask', { reason: 'The torso is fully covered by armor.' }),
    },
    segment,
  });
  expect(result).toMatchObject({
    status: 'no-detections',
    finishReason: 'llm-confirmed-absence',
    llmCalls: 1,
    samCalls: 0,
    boxes: [],
  });
  expect(segment).not.toHaveBeenCalled();
});

it('treats truncated output as incomplete, even when it contains a syntactically valid absence report', async () => {
  const { options } = await setup();
  const result = await runSamAgent(options, {
    llm: { complete: async () => reply('report_no_mask', { reason: 'No match.' }, 'length') },
  });
  expect(result).toMatchObject({ status: 'incomplete', finishReason: 'output-limit', llmCalls: 1 });
});

it('rejects invalid final selections while preserving the latest provisional candidates', async () => {
  const { options, first } = await setup();
  const responses = [
    reply('segment_phrase', { text_prompt: 'bare chest', reason: 'Find bare skin.' }),
    reply('segment_phrase', { text_prompt: 'bare chest', reason: 'Repeat.' }),
    reply('select_masks_and_return', { mask_ids: [2], reason: 'Pick candidate.' }),
  ];
  const complete = vi.fn(async () => responses[complete.mock.calls.length - 1]!);
  const segment = vi.fn(async () => first);
  const result = await runSamAgent(options, { llm: { complete }, segment });
  expect(result).toMatchObject({
    status: 'incomplete',
    boxes: first.boxes,
    scores: first.scores,
    boxSource: 'unreviewed-candidates',
    llmCalls: 3,
    samCalls: 1,
  });
  expect(segment).toHaveBeenCalledTimes(1);
});

it('does not silently reuse cached decisions after an environment-selected model change', async () => {
  const { options } = await setup();
  await runSamAgent(options, {
    llm: { complete: async () => reply('report_no_mask', { reason: 'Covered.' }) },
  });
  vi.stubEnv('SAM_AGENT_MODEL', 'google/gemini-3.5-flash-lite');
  const complete = vi.fn();
  await expect(
    runSamAgent({ ...options, settings: agentSettingsFromEnv() }, { llm: { complete } }),
  ).rejects.toThrow('conditions-mismatch');
  expect(complete).not.toHaveBeenCalled();
});

it('does not resend a paid or uncertain request after an interruption or service error', async () => {
  const { options } = await setup();
  const complete = vi.fn(async () => {
    throw new Error('openrouter-http-402');
  });
  await expect(runSamAgent(options, { llm: { complete } })).rejects.toThrow('openrouter-http-402');
  await expect(runSamAgent(options, { llm: { complete } })).rejects.toThrow('outcome-unknown');
  expect(complete).toHaveBeenCalledTimes(1);
});

it('transport uses the selected model, image input and exact output cap without leaking secrets or retrying errors', async () => {
  const { options } = await setup();
  vi.stubEnv('SAM_AGENT_MODEL', 'test/vision-model');
  const perform = vi.fn(async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    expect(body.model).toBe('test/vision-model');
    expect(body.max_tokens).toBe(1024);
    expect(JSON.stringify(body.messages)).toContain('data:image/jpeg;base64,');
    expect(init?.headers).toMatchObject({ Authorization: 'Bearer test-only-key' });
    return new Response(JSON.stringify(reply('report_no_mask', { reason: 'Covered.' })), {
      status: 200,
    });
  }) as ReturnType<typeof vi.fn<typeof fetch>>;
  await runSamAgent(
    { ...options, settings: agentSettingsFromEnv() },
    { llm: new OpenRouterClient('test-only-key', perform) },
  );
  expect(perform).toHaveBeenCalledTimes(1);
  const failing = vi.fn(async () => new Response('test-only-key sensitive error', { status: 402 }));
  const client = new OpenRouterClient('test-only-key', failing);
  await expect(client.complete({} as LlmRequest)).rejects.toThrow('openrouter-http-402');
  expect(failing).toHaveBeenCalledTimes(1);
});

it('uses the real SAM adapter to show numbered mask candidates and maps selected IDs to the correct boxes', async () => {
  const { options } = await setup();
  vi.mocked(runSamBridge).mockImplementation(async (o) => {
    await mkdir(o.out, { recursive: true });
    const png_masks: string[] = [];
    for (const [index, offset] of [28, 168].entries()) {
      const path = join(o.out, `mask-${index}.png`);
      const pixels = Buffer.alloc(280 * 280);
      for (let y = offset; y < offset + 56; y++)
        for (let x = offset; x < offset + 56; x++) pixels[y * 280 + x] = 255;
      await sharp(pixels, { raw: { width: 280, height: 280, channels: 1 } })
        .png()
        .toFile(path);
      png_masks.push(path);
    }
    return {
      original_image_path: o.image,
      orig_img_w: 280,
      orig_img_h: 280,
      pred_boxes: [
        [0.1, 0.1, 0.2, 0.2],
        [0.6, 0.6, 0.2, 0.2],
      ],
      pred_scores: [0.9, 0.8],
      png_masks,
      output_image_path: options.image,
      requestId: 'sam-mock',
      elapsedMs: 1,
      prompt: o.prompt,
      recoveredFromSavedResponse: false,
    };
  });
  let round = 0;
  const complete = vi.fn(async (r: LlmRequest) => {
    if (++round === 1)
      return reply('segment_phrase', {
        text_prompt: 'bare torso',
        reason: 'Locate the exposed torso.',
      });
    const user = r.messages[1]!.content;
    const text = Array.isArray(user)
      ? user
          .filter((item) => item.type === 'text')
          .map((item) => item.text)
          .join('\n')
      : user;
    expect(text).toContain('"id":2');
    expect(Array.isArray(user) && user.filter((item) => item.type === 'image_url').length).toBe(2);
    return reply('select_masks_and_return', {
      mask_ids: [2],
      reason: 'Only the second region matches.',
    });
  });
  const result = await runSamAgent(options, { llm: { complete } });
  expect(result).toMatchObject({
    status: 'detected',
    boxes: [{ x0: 168, y0: 168, x1: 224, y1: 224 }],
    scores: [0.8],
    llmCalls: 2,
    samCalls: 1,
  });
  const candidates = JSON.parse(
    await readFile(join(options.out, 'turn-01.candidates.json'), 'utf8'),
  );
  expect((await sharp(await readFile(candidates.overlayPath)).metadata()).width).toBe(280);
});

it('defaults to Qwen3.7 Flash when the optional model environment variable is absent', () => {
  vi.stubEnv('SAM_AGENT_MODEL', undefined);
  expect(agentSettingsFromEnv().model).toBe('qwen/qwen3.7-flash');
  vi.stubEnv('SAM_AGENT_MODEL', '');
  expect(() => agentSettingsFromEnv()).toThrow('Invalid SAM_AGENT_MODEL');
});

it('preserves candidate boxes when the reviewing response hits the output token limit', async () => {
  const { options, first } = await setup();
  let turn = 0;
  const result = await runSamAgent(options, {
    llm: {
      complete: async () =>
        ++turn === 1
          ? reply('segment_phrase', {
              text_prompt: 'bare chest',
              reason: 'Locate the visible target.',
            })
          : reply('report_no_mask', { reason: 'Incomplete reasoning.' }, 'length'),
    },
    segment: async () => first,
  });
  expect(result).toMatchObject({
    status: 'incomplete',
    finishReason: 'output-limit',
    boxSource: 'unreviewed-candidates',
    boxes: first.boxes,
    scores: first.scores,
    llmCalls: 2,
  });
});

it('does not resurrect earlier candidates when the latest query has no detections', async () => {
  const { options, first } = await setup();
  let turn = 0;
  const complete = vi.fn(async () =>
    reply('segment_phrase', { text_prompt: `phrase ${++turn}`, reason: 'Try a target.' }),
  );
  const segment = vi.fn(async () =>
    segment.mock.calls.length === 1 ? first : { ...first, boxes: [], scores: [] },
  );
  const result = await runSamAgent(options, { llm: { complete }, segment });
  expect(result).toMatchObject({
    status: 'incomplete',
    boxSource: 'none',
    boxes: [],
    scores: [],
    samCalls: 2,
  });
});

it('discards provisional boxes when the LLM explicitly confirms that the image has no matching region', async () => {
  const { options, first } = await setup();
  let turn = 0;
  const result = await runSamAgent(options, {
    llm: {
      complete: async () =>
        ++turn === 1
          ? reply('segment_phrase', { text_prompt: 'bare chest', reason: 'Locate the chest.' })
          : reply('report_no_mask', { reason: 'The candidate is armor, not visible skin.' }),
    },
    segment: async () => first,
  });
  expect(result).toMatchObject({
    status: 'no-detections',
    boxSource: 'none',
    boxes: [],
    llmCalls: 2,
  });
});
