/**
 * Anthropic provider and LLM telemetry helpers.
 */
import Anthropic from "@anthropic-ai/sdk";
import type {
  MessageCreateParamsNonStreaming,
  Message,
} from "@anthropic-ai/sdk/resources/messages";
import { createLogger } from "../../logger";
import { randomUUID } from "crypto";
import type { LlmAuthMode, LlmProviderId } from "../types";

const log = createLogger("anthropic");

const PRICING: Record<
  string,
  { input: number; output: number; cacheRead: number; cacheWrite: number }
> = {
  "claude-opus-4-20250514": { input: 15.0, output: 75.0, cacheRead: 1.5, cacheWrite: 18.75 },
  "claude-opus-4-6": { input: 15.0, output: 75.0, cacheRead: 1.5, cacheWrite: 18.75 },
  "claude-sonnet-4-20250514": { input: 3.0, output: 15.0, cacheRead: 0.3, cacheWrite: 3.75 },
  "claude-sonnet-4-5-20250929": { input: 3.0, output: 15.0, cacheRead: 0.3, cacheWrite: 3.75 },
  "claude-haiku-4-5-20251001": { input: 0.8, output: 4.0, cacheRead: 0.08, cacheWrite: 1.0 },
  "claude-3-5-sonnet-20241022": { input: 3.0, output: 15.0, cacheRead: 0.3, cacheWrite: 3.75 },
  "claude-3-5-haiku-20241022": { input: 0.8, output: 4.0, cacheRead: 0.08, cacheWrite: 1.0 },
};

const DEFAULT_PRICING = { input: 3.0, output: 15.0, cacheRead: 0.3, cacheWrite: 3.75 };

interface RetryConfig {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
}

const RETRY_CONFIGS: Record<string, RetryConfig> = {
  rate_limit: { maxRetries: 5, initialDelayMs: 1000, maxDelayMs: 30000 },
  server_error: { maxRetries: 3, initialDelayMs: 2000, maxDelayMs: 30000 },
  connection: { maxRetries: 3, initialDelayMs: 1000, maxDelayMs: 10000 },
};

export interface LlmCallRecord {
  id: string;
  created_at: string;
  model: string;
  caller: string;
  email_id: string | null;
  account_id: string | null;
  provider_id: LlmProviderId;
  auth_mode: LlmAuthMode | null;
  fallback_used: number;
  cost_estimated: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_create_tokens: number;
  cost_cents: number;
  duration_ms: number;
  success: number;
  error_message: string | null;
}

export interface UsageStats {
  today: { totalCostCents: number; totalCalls: number };
  thisWeek: { totalCostCents: number; totalCalls: number };
  thisMonth: { totalCostCents: number; totalCalls: number };
  byModel: Array<{ model: string; costCents: number; calls: number }>;
  byCaller: Array<{ caller: string; costCents: number; calls: number }>;
}

export interface CreateOptions {
  caller: string;
  emailId?: string;
  accountId?: string;
  timeoutMs?: number;
}

export type DatabaseInstance = {
  prepare: (sql: string) => {
    run: (...args: unknown[]) => void;
    get: (...args: unknown[]) => unknown;
    all: (...args: unknown[]) => unknown[];
  };
  exec: (sql: string) => void;
  transaction: <T>(fn: () => T) => () => T;
};

let _anthropicClient: Anthropic | null = null;
let _defaultClient: Anthropic | null = null;
let _db: DatabaseInstance | null = null;
let _insertStmt: ReturnType<DatabaseInstance["prepare"]> | null = null;

export function hasAnthropicCredentials(apiKey?: string): boolean {
  return Boolean(_anthropicClient || process.env.ANTHROPIC_API_KEY || apiKey);
}

export function _setClientForTesting(client: unknown): void {
  _anthropicClient = client as Anthropic;
}

export function resetClient(): void {
  _defaultClient = null;
}

export function getClient(): Anthropic {
  if (_anthropicClient) return _anthropicClient;
  if (!_defaultClient) _defaultClient = new Anthropic();
  return _defaultClient;
}

export function setAnthropicServiceDb(db: DatabaseInstance): void {
  _db = db;
  db.exec(`
    CREATE TABLE IF NOT EXISTS llm_calls (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      model TEXT NOT NULL,
      caller TEXT NOT NULL,
      email_id TEXT,
      account_id TEXT,
      provider_id TEXT NOT NULL DEFAULT 'anthropic',
      auth_mode TEXT,
      fallback_used INTEGER NOT NULL DEFAULT 0,
      cost_estimated INTEGER NOT NULL DEFAULT 0,
      input_tokens INTEGER NOT NULL,
      output_tokens INTEGER NOT NULL,
      cache_read_tokens INTEGER DEFAULT 0,
      cache_create_tokens INTEGER DEFAULT 0,
      cost_cents REAL NOT NULL,
      duration_ms INTEGER NOT NULL,
      success INTEGER NOT NULL DEFAULT 1,
      error_message TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at);
    CREATE INDEX IF NOT EXISTS idx_llm_calls_caller ON llm_calls(caller);
  `);
  _insertStmt = db.prepare(`
    INSERT INTO llm_calls (id, model, caller, email_id, account_id,
      provider_id, auth_mode, fallback_used, cost_estimated,
      input_tokens, output_tokens, cache_read_tokens, cache_create_tokens,
      cost_cents, duration_ms, success, error_message)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
}

export function calculateCostCents(
  model: string,
  inputTokens: number,
  outputTokens: number,
  cacheReadTokens: number,
  cacheCreateTokens: number,
): number {
  const pricing = PRICING[model] || DEFAULT_PRICING;
  const inputCost = (inputTokens * pricing.input) / 1_000_000;
  const outputCost = (outputTokens * pricing.output) / 1_000_000;
  const cacheReadCost = (cacheReadTokens * pricing.cacheRead) / 1_000_000;
  const cacheWriteCost = (cacheCreateTokens * pricing.cacheWrite) / 1_000_000;
  return (inputCost + outputCost + cacheReadCost + cacheWriteCost) * 100;
}

export function recordLlmCall(params: {
  model: string;
  caller: string;
  emailId: string | null;
  accountId: string | null;
  usage: Record<string, number>;
  durationMs: number;
  success: boolean;
  errorMessage: string | null;
  providerId?: LlmProviderId;
  authMode?: LlmAuthMode | null;
  fallbackUsed?: boolean;
  costCentsOverride?: number;
  costEstimated?: boolean;
}): void {
  if (!_insertStmt) {
    log.warn("AnthropicService: database not initialized, skipping call recording");
    return;
  }

  const inputTokens = params.usage.input_tokens || 0;
  const outputTokens = params.usage.output_tokens || 0;
  const cacheReadTokens = params.usage.cache_read_input_tokens || 0;
  const cacheCreateTokens = params.usage.cache_creation_input_tokens || 0;
  const costCents =
    params.costCentsOverride ??
    calculateCostCents(
      params.model,
      inputTokens,
      outputTokens,
      cacheReadTokens,
      cacheCreateTokens,
    );

  try {
    _insertStmt.run(
      randomUUID(),
      params.model,
      params.caller,
      params.emailId,
      params.accountId,
      params.providerId ?? "anthropic",
      params.authMode ?? "unknown",
      params.fallbackUsed ? 1 : 0,
      params.costEstimated ? 1 : 0,
      inputTokens,
      outputTokens,
      cacheReadTokens,
      cacheCreateTokens,
      costCents,
      params.durationMs,
      params.success ? 1 : 0,
      params.errorMessage,
    );
  } catch (err) {
    log.error({ err }, "Failed to record LLM call to database");
  }
}

export function recordStreamingCall(
  model: string,
  caller: string,
  usage: Record<string, number>,
  durationMs: number,
  options?: {
    emailId?: string;
    accountId?: string;
    providerId?: LlmProviderId;
    authMode?: LlmAuthMode | null;
    fallbackUsed?: boolean;
    costEstimated?: boolean;
  },
): void {
  recordLlmCall({
    model,
    caller,
    emailId: options?.emailId || null,
    accountId: options?.accountId || null,
    usage,
    durationMs,
    success: true,
    errorMessage: null,
    providerId: options?.providerId,
    authMode: options?.authMode,
    fallbackUsed: options?.fallbackUsed,
    costEstimated: options?.costEstimated,
  });
}

function asyncSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getRetryCategory(error: unknown): string | null {
  if (error instanceof Anthropic.RateLimitError) return "rate_limit";
  if (error instanceof Anthropic.InternalServerError) return "server_error";
  if (error instanceof Anthropic.APIConnectionError) return "connection";
  if (error instanceof Anthropic.APIError && (error as { status?: number }).status === 529) {
    return "server_error";
  }
  return null;
}

export async function createAnthropicMessage(
  params: MessageCreateParamsNonStreaming,
  options: CreateOptions,
): Promise<Message> {
  const { caller, emailId, accountId, timeoutMs } = options;
  const model = params.model;
  const startTime = Date.now();

  const client = getClient();
  let lastError: unknown = null;
  let totalAttempts = 0;
  const maxPossibleRetries = Math.max(...Object.values(RETRY_CONFIGS).map((c) => c.maxRetries));

  for (let attempt = 0; attempt <= maxPossibleRetries; attempt++) {
    totalAttempts = attempt + 1;

    let abortController: AbortController | undefined;
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    if (timeoutMs) {
      abortController = new AbortController();
      timeoutHandle = setTimeout(() => abortController!.abort(), timeoutMs);
    }

    try {
      const response = await client.messages.create(params, {
        signal: abortController?.signal,
      });

      const usage = response.usage as unknown as Record<string, number>;
      recordLlmCall({
        model,
        caller,
        emailId: emailId || null,
        accountId: accountId || null,
        usage,
        durationMs: Date.now() - startTime,
        success: true,
        errorMessage: null,
        providerId: "anthropic",
        authMode: "api_key",
      });

      if (totalAttempts > 1) {
        log.info({ caller, model, attempts: totalAttempts }, "LLM call succeeded after retries");
      }

      return response;
    } catch (error) {
      lastError = error;
      const category = getRetryCategory(error);

      if (!category) {
        break;
      }

      const config = RETRY_CONFIGS[category];
      if (attempt >= config.maxRetries) {
        break;
      }

      const baseDelay = Math.min(config.initialDelayMs * Math.pow(2, attempt), config.maxDelayMs);
      const jitter = baseDelay * 0.1 * Math.random();
      const delay = baseDelay + jitter;

      log.warn(
        {
          caller,
          model,
          attempt: attempt + 1,
          maxRetries: config.maxRetries,
          category,
          delayMs: Math.round(delay),
        },
        "LLM call failed, retrying",
      );

      await asyncSleep(delay);
    } finally {
      if (timeoutHandle) clearTimeout(timeoutHandle);
    }
  }

  const errMsg = lastError instanceof Error ? lastError.message : String(lastError);
  recordLlmCall({
    model,
    caller,
    emailId: emailId || null,
    accountId: accountId || null,
    usage: {},
    durationMs: Date.now() - startTime,
    success: false,
    errorMessage: errMsg,
    providerId: "anthropic",
    authMode: "api_key",
  });

  throw lastError;
}

export function getUsageStats(): UsageStats {
  if (!_db) {
    return {
      today: { totalCostCents: 0, totalCalls: 0 },
      thisWeek: { totalCostCents: 0, totalCalls: 0 },
      thisMonth: { totalCostCents: 0, totalCalls: 0 },
      byModel: [],
      byCaller: [],
    };
  }

  const today = _db
    .prepare(
      "SELECT COALESCE(SUM(cost_cents), 0) as cost, COUNT(*) as calls FROM llm_calls WHERE date(created_at) = date('now')",
    )
    .get() as { cost: number; calls: number };

  const thisWeek = _db
    .prepare(
      "SELECT COALESCE(SUM(cost_cents), 0) as cost, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-7 days')",
    )
    .get() as { cost: number; calls: number };

  const thisMonth = _db
    .prepare(
      "SELECT COALESCE(SUM(cost_cents), 0) as cost, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-30 days')",
    )
    .get() as { cost: number; calls: number };

  const byModel = _db
    .prepare(
      "SELECT model, COALESCE(SUM(cost_cents), 0) as costCents, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-30 days') GROUP BY model ORDER BY costCents DESC",
    )
    .all() as Array<{ model: string; costCents: number; calls: number }>;

  const byCaller = _db
    .prepare(
      "SELECT caller, COALESCE(SUM(cost_cents), 0) as costCents, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-30 days') GROUP BY caller ORDER BY costCents DESC",
    )
    .all() as Array<{ caller: string; costCents: number; calls: number }>;

  return {
    today: { totalCostCents: today.cost, totalCalls: today.calls },
    thisWeek: { totalCostCents: thisWeek.cost, totalCalls: thisWeek.calls },
    thisMonth: { totalCostCents: thisMonth.cost, totalCalls: thisMonth.calls },
    byModel,
    byCaller,
  };
}

export function getCallHistory(limit: number = 50): LlmCallRecord[] {
  if (!_db) return [];

  return _db
    .prepare("SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT ?")
    .all(limit) as LlmCallRecord[];
}
