import { query } from "@anthropic-ai/claude-agent-sdk";
import type { Message, MessageCreateParamsNonStreaming } from "@anthropic-ai/sdk/resources/messages";
import { randomUUID } from "crypto";
import type { Config } from "../../../../shared/types";
import { createLogger } from "../../logger";
import { buildClaudeSdkEnv, resolvedClaudeSdkCliPath, spawnClaudeSdkProcess } from "../claude-sdk-runtime";
import type {
  LlmExecutionRequest,
  LlmProvider,
  LlmProviderAvailability,
  LlmProviderResult,
} from "../types";

const log = createLogger("llm-claude-sdk");

const CAPABILITIES = ["basic-text", "structured-output"] as const;
const PROBE_TTL_MS = 60_000;

type ProbeCache = {
  expiresAt: number;
  value: LlmProviderAvailability;
};

let probeCache: ProbeCache | null = null;

export class ClaudeSdkAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ClaudeSdkAuthError";
  }
}

export function determineClaudeSdkAuthMode(params: {
  apiKeyConfigured: boolean;
  account?: { apiKeySource?: string; tokenSource?: string } | null;
}): "api_key" | "claude_login" | "unknown" {
  if (params.apiKeyConfigured || params.account?.apiKeySource) return "api_key";
  if (params.account?.tokenSource) return "claude_login";
  return "unknown";
}

function looksLikeClaudeSdkAuthError(message: string): boolean {
  const normalized = message.toLowerCase();
  return normalized.includes("not logged in") || normalized.includes("/login") || normalized.includes("auth");
}

function messageContentToText(content: MessageCreateParamsNonStreaming["messages"][number]["content"]): string {
  if (typeof content === "string") return content;
  return content
    .map((block) => {
      if (block.type === "text") return block.text;
      return `[${block.type}]`;
    })
    .join("\n\n");
}

function systemToText(system: MessageCreateParamsNonStreaming["system"]): string {
  if (!system) return "";
  if (typeof system === "string") return system;
  return system
    .map((block) => (block.type === "text" ? block.text : ""))
    .filter(Boolean)
    .join("\n\n");
}

function buildPrompt(params: MessageCreateParamsNonStreaming): { prompt: string; systemPrompt?: string } {
  const systemPrompt = systemToText(params.system);
  const prompt = params.messages
    .map((message) => `${message.role.toUpperCase()}:\n${messageContentToText(message.content)}`)
    .join("\n\n");

  return {
    prompt,
    systemPrompt: systemPrompt || undefined,
  };
}

function makeCompatibleMessage(model: string, text: string, usage: Record<string, number>): Message {
  return {
    id: `msg_sdk_${randomUUID()}`,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model,
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: usage.input_tokens ?? 0,
      output_tokens: usage.output_tokens ?? 0,
      cache_read_input_tokens: usage.cache_read_input_tokens ?? 0,
      cache_creation_input_tokens: usage.cache_creation_input_tokens ?? 0,
    },
  };
}

async function runAvailabilityProbe(model: string, timeoutMs: number): Promise<LlmProviderAvailability> {
  const abortController = new AbortController();
  const timeoutHandle = setTimeout(() => abortController.abort(), timeoutMs);
  const q = query({
    prompt: "Respond with OK.",
    options: {
      model,
      abortController,
      permissionMode: "dontAsk",
      persistSession: false,
      env: buildClaudeSdkEnv(),
      pathToClaudeCodeExecutable: resolvedClaudeSdkCliPath,
      spawnClaudeCodeProcess: spawnClaudeSdkProcess,
      allowedTools: [],
    },
  });

  try {
    const account = await q.accountInfo();
    return {
      providerId: "claude-sdk",
      available: true,
      authMode: determineClaudeSdkAuthMode({
        apiKeyConfigured: Boolean(process.env.ANTHROPIC_API_KEY),
        account,
      }),
      capabilities: [...CAPABILITIES],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log.info({ err: message }, "Claude SDK availability probe failed");
    return {
      providerId: "claude-sdk",
      available: false,
      authMode: "unknown",
      reason: message,
      capabilities: [...CAPABILITIES],
    };
  } finally {
    clearTimeout(timeoutHandle);
    q.close();
  }
}

export function resetClaudeSdkProbeCache(): void {
  probeCache = null;
}

export const claudeSdkProvider: LlmProvider = {
  id: "claude-sdk",
  capabilities: [...CAPABILITIES],
  async isAvailable(config: Config): Promise<LlmProviderAvailability> {
    if (probeCache && probeCache.expiresAt > Date.now()) {
      return probeCache.value;
    }

    const value = await runAvailabilityProbe(config.model, 7_500);
    probeCache = { value, expiresAt: Date.now() + PROBE_TTL_MS };
    return value;
  },
  async execute(request: LlmExecutionRequest): Promise<LlmProviderResult> {
    const { prompt, systemPrompt } = buildPrompt(request.params);
    const abortController = new AbortController();
    const timeoutHandle = request.options.timeoutMs
      ? setTimeout(() => abortController.abort(), request.options.timeoutMs)
      : undefined;
    const startTime = Date.now();

    const q = query({
      prompt,
      options: {
        model: request.params.model,
        abortController,
        permissionMode: "dontAsk",
        persistSession: false,
        maxTurns: 1,
        env: buildClaudeSdkEnv(),
        pathToClaudeCodeExecutable: resolvedClaudeSdkCliPath,
        spawnClaudeCodeProcess: spawnClaudeSdkProcess,
        allowedTools: [],
        systemPrompt,
      },
    });

    try {
      for await (const message of q) {
        if (message.type !== "result") continue;
        if (message.subtype !== "success") {
          const errorMessage =
            message.errors?.join("\n") || `Claude SDK request failed: ${message.subtype}`;
          if (looksLikeClaudeSdkAuthError(errorMessage)) {
            throw new ClaudeSdkAuthError(errorMessage);
          }
          throw new Error(errorMessage);
        }
        if (looksLikeClaudeSdkAuthError(message.result)) {
          throw new ClaudeSdkAuthError(message.result);
        }

        const usage = {
          input_tokens: message.usage.inputTokens,
          output_tokens: message.usage.outputTokens,
          cache_read_input_tokens: message.usage.cacheReadInputTokens,
          cache_creation_input_tokens: message.usage.cacheCreationInputTokens,
        };

        return {
          message: makeCompatibleMessage(request.params.model, message.result, usage),
          providerId: "claude-sdk",
          authMode: determineClaudeSdkAuthMode({
            apiKeyConfigured: Boolean(process.env.ANTHROPIC_API_KEY || request.config.anthropicApiKey),
          }),
          usage,
          durationMs: Date.now() - startTime,
          costCents: typeof message.total_cost_usd === "number" ? message.total_cost_usd * 100 : undefined,
          costEstimated: false,
        };
      }

      throw new Error("Claude SDK returned no result message");
    } finally {
      if (timeoutHandle) clearTimeout(timeoutHandle);
      q.close();
    }
  },
};
