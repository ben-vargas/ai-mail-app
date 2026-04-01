/**
 * Compatibility wrapper for internal LLM calls.
 * Existing feature code keeps importing this module while the actual provider
 * routing lives in src/main/services/llm/.
 */
import type { MessageCreateParamsNonStreaming, Message } from "@anthropic-ai/sdk/resources/messages";
import { ConfigSchema } from "../../shared/types";
import { createCompatibilityMessage } from "./llm/facade";
import { createLogger } from "./logger";
import {
  _setClientForTesting,
  createAnthropicMessage,
  getCallHistory,
  getClient,
  getUsageStats,
  recordStreamingCall,
  resetClient,
  setAnthropicServiceDb,
  type CreateOptions,
  type LlmCallRecord,
  type UsageStats,
} from "./llm/providers/anthropic-provider";
import { resetLlmReadinessCaches } from "./llm/readiness";

export {
  _setClientForTesting,
  createAnthropicMessage,
  getCallHistory,
  getClient,
  getUsageStats,
  recordStreamingCall,
  resetClient,
  setAnthropicServiceDb,
};
export type { CreateOptions, LlmCallRecord, UsageStats };

const log = createLogger("anthropic-service");

export async function createMessage(
  params: MessageCreateParamsNonStreaming,
  options: CreateOptions,
): Promise<Message> {
  try {
    const { getConfig } = await import("../ipc/settings.ipc");
    return createCompatibilityMessage(params, options, getConfig());
  } catch {
    log.warn("Failed to load settings config, using schema defaults for LLM routing");
    return createCompatibilityMessage(params, options, ConfigSchema.parse({}));
  }
}

export function resetInternalLlmState(): void {
  resetClient();
  resetLlmReadinessCaches();
}
