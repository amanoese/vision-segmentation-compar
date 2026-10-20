import { createHash } from 'node:crypto';
import { SYSTEM_PROMPT } from './prompt.js';
import { z } from 'zod';

export const DEFAULT_AGENT_MODEL = 'qwen/qwen3.7-flash';
export const AGENT_PROMPT_VERSION = 'bounded-visual-agent-v2';
export function agentSettingsFromEnv() {
  const model = process.env.SAM_AGENT_MODEL ?? DEFAULT_AGENT_MODEL;
  if (!model.trim() || model !== model.trim()) throw new Error('Invalid SAM_AGENT_MODEL');
  return {
    model,
    maxLlmCalls: 3 as const,
    maxOutputTokens: 1024,
    reasoningEnabled: false,
    promptVersion: AGENT_PROMPT_VERSION,
    promptSha256: createHash('sha256').update(SYSTEM_PROMPT).digest('hex'),
  };
}
export type AgentSettings = ReturnType<typeof agentSettingsFromEnv>;
export type Message = {
  role: 'system' | 'user';
  content:
    | string
    | Array<{ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }>;
};
export interface LlmRequest {
  model: string;
  messages: Message[];
  tools: Array<{
    type: 'function';
    function: { name: string; description: string; parameters: object };
  }>;
  tool_choice: 'required';
  parallel_tool_calls: false;
  max_tokens: number;
  reasoning: { enabled: boolean };
}
const replySchema = z.object({
  id: z.string().optional(),
  model: z.string(),
  choices: z
    .array(
      z.object({
        finish_reason: z.string().nullable(),
        message: z.object({
          content: z.string().nullable().optional(),
          tool_calls: z
            .array(
              z.object({
                id: z.string(),
                type: z.literal('function'),
                function: z.object({ name: z.string(), arguments: z.string() }),
              }),
            )
            .optional(),
        }),
      }),
    )
    .min(1),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
      cost: z.number().optional(),
    })
    .optional(),
});
export type LlmReply = z.infer<typeof replySchema>;
export interface LlmClient {
  complete(request: LlmRequest): Promise<LlmReply>;
}
export class OpenRouterClient implements LlmClient {
  constructor(
    private readonly apiKey: string,
    private readonly perform: typeof fetch = globalThis.fetch,
  ) {
    if (!apiKey) throw new Error('openrouter-key-missing');
  }
  async complete(request: LlmRequest): Promise<LlmReply> {
    let response: Response;
    try {
      response = await this.perform('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        redirect: 'error',
        headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(60_000),
      });
    } catch {
      throw new Error('openrouter-network-failed-or-timeout');
    }
    // Never include response bodies or credentials in error messages; never retry POSTs.
    if (!response.ok) throw new Error(`openrouter-http-${response.status}`);
    try {
      return replySchema.parse(await response.json());
    } catch {
      throw new Error('openrouter-invalid-response');
    }
  }
}

export function requestWithoutImages(request: LlmRequest) {
  return {
    ...request,
    messages: request.messages.map((message) => ({
      ...message,
      content:
        typeof message.content === 'string'
          ? message.content
          : message.content.map((item) =>
              item.type === 'image_url'
                ? { type: 'image_url', image_url: { url: '[local image omitted]' } }
                : item,
            ),
    })),
  };
}
