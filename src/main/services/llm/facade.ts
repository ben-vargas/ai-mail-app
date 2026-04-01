import type { Message } from "@anthropic-ai/sdk/resources/messages";
import type { Config } from "../../../shared/types";
import { createLogger } from "../logger";
import {
  createAnthropicMessage,
  hasAnthropicCredentials,
  recordLlmCall,
  type CreateOptions,
} from "./providers/anthropic-provider";
import {
  ClaudeSdkAuthError,
  claudeSdkProvider,
  determineClaudeSdkAuthMode,
} from "./providers/claude-sdk-provider";
import { decideRoute, extractRequestFeatures } from "./provider-policy";
import type { LlmExecutionRequest, LlmProviderResult } from "./types";

const log = createLogger("llm-facade");

function shouldRecordFallback(attemptIndex: number): boolean {
  return attemptIndex > 0;
}

function shouldAttemptFallbackAfterError(error: unknown): boolean {
  return error instanceof ClaudeSdkAuthError;
}

async function isProviderAvailable(providerId: "anthropic" | "claude-sdk", config: Config): Promise<boolean> {
  if (providerId === "anthropic") {
    return hasAnthropicCredentials(config.anthropicApiKey);
  }
  const availability = await claudeSdkProvider.isAvailable(config);
  return availability.available;
}

export async function createCompatibilityMessage(
  params: Parameters<typeof createAnthropicMessage>[0],
  options: CreateOptions,
  config: Config,
): Promise<Message> {
  const request: LlmExecutionRequest = {
    params,
    options,
    config,
    features: extractRequestFeatures({ params }),
  };
  const route = decideRoute(request);
  let lastError: unknown = null;

  for (const [attemptIndex, providerId] of route.attemptedProviderIds.entries()) {
    if (!(await isProviderAvailable(providerId, config))) {
      continue;
    }
    try {
      if (providerId === "anthropic") {
        return await createAnthropicMessage(params, options);
      }

      const result: LlmProviderResult = await claudeSdkProvider.execute(request);
      recordLlmCall({
        model: params.model,
        caller: options.caller,
        emailId: options.emailId ?? null,
        accountId: options.accountId ?? null,
        usage: result.usage,
        durationMs: result.durationMs,
        success: true,
        errorMessage: null,
        providerId: result.providerId,
        authMode: result.authMode,
        fallbackUsed: shouldRecordFallback(attemptIndex),
        costCentsOverride: result.costCents,
        costEstimated: result.costEstimated ?? false,
      });
      return result.message;
    } catch (error) {
      lastError = error;
      log.warn(
        {
          err: error instanceof Error ? error.message : String(error),
          providerId,
          caller: options.caller,
          model: params.model,
        },
        "LLM provider attempt failed",
      );
      if (providerId === "claude-sdk") {
        recordLlmCall({
          model: params.model,
          caller: options.caller,
          emailId: options.emailId ?? null,
          accountId: options.accountId ?? null,
          usage: {},
          durationMs: 0,
          success: false,
          errorMessage: error instanceof Error ? error.message : String(error),
          providerId: "claude-sdk",
          authMode: determineClaudeSdkAuthMode({
            apiKeyConfigured: Boolean(process.env.ANTHROPIC_API_KEY || config.anthropicApiKey),
          }),
          fallbackUsed: shouldRecordFallback(attemptIndex),
          costEstimated: false,
        });
      }
      if (!shouldAttemptFallbackAfterError(error)) {
        throw error;
      }
      continue;
    }
  }

  throw lastError instanceof Error ? lastError : new Error("No LLM provider could satisfy the request");
}
