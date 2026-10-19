import { createFalClient } from '@fal-ai/client';
import { setTimeout as delay } from 'node:timers/promises';

export const ENDPOINT = 'fal-ai/sam-3-1/image' as const;
export type Input = { image_url: string; prompt: string; [key: string]: unknown };
export interface Inference {
  infer(
    input: Input,
    onEnqueue: (id: string) => Promise<void>,
  ): Promise<{ data: unknown; requestId: string; elapsedMs: number }>;
}

// AbortError also prevents the SDK's queue-specific retry policy from resending POSTs.
class SafeFailure extends Error {
  override name = 'AbortError';
  constructor(
    code: string,
    public readonly status?: number,
  ) {
    super(code);
  }
}
export function safeError(error: unknown): string {
  return error instanceof SafeFailure ? error.message : 'processing-failed';
}

export class SamTransport implements Inference {
  requests = 0;
  stopped = false;
  constructor(
    private readonly credentials: string,
    private readonly options: { fetch?: typeof fetch; timeoutMs?: number; pollMs?: number } = {},
  ) {}
  async infer(input: Input, onEnqueue: (id: string) => Promise<void>) {
    if (this.stopped) throw new SafeFailure('backend-stopped');
    if (this.requests >= 5) throw new SafeFailure('request-budget-exhausted');
    this.requests++;
    const started = Date.now(),
      signal = AbortSignal.timeout(this.options.timeoutMs ?? 120_000);
    let submitted = false,
      requestId: string | undefined;
    const perform = this.options.fetch ?? globalThis.fetch;
    const guardedFetch: typeof fetch = async (url, init) => {
      if (init?.method?.toUpperCase() === 'POST') {
        if (submitted) throw new SafeFailure('submission-retry-disabled');
        submitted = true;
      }
      try {
        const response = await perform(url, init);
        if (!response.ok) throw new SafeFailure(`fal-http-${response.status}`, response.status);
        return response;
      } catch (error) {
        if (error instanceof SafeFailure) throw error;
        throw new SafeFailure(signal.aborted ? 'fal-timeout' : 'fal-network-failed');
      }
    };
    const client = createFalClient({
      credentials: this.credentials,
      retry: { maxRetries: 0 },
      fetch: guardedFetch,
    });
    try {
      const queued = await client.queue.submit(ENDPOINT, { input, abortSignal: signal });
      requestId = queued.request_id;
      await onEnqueue(requestId);
      for (;;) {
        const state = await client.queue.status(ENDPOINT, { requestId, abortSignal: signal });
        if (state.status === 'COMPLETED') break;
        await delay(this.options.pollMs ?? 750, undefined, { signal });
      }
      const result = await client.queue.result(ENDPOINT, { requestId, abortSignal: signal });
      return { data: result.data as unknown, requestId, elapsedMs: Date.now() - started };
    } catch (error) {
      if (error instanceof SafeFailure && [401, 402, 403].includes(error.status ?? 0))
        this.stopped = true;
      if (signal.aborted) {
        if (requestId) {
          try {
            await client.queue.cancel(ENDPOINT, {
              requestId,
              abortSignal: AbortSignal.timeout(3000),
            });
          } catch {
            /* An interrupted job can still incur a charge; do not resubmit. */
          }
        }
        throw new SafeFailure('fal-timeout');
      }
      throw error;
    }
  }
}
