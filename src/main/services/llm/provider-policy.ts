import type { Config, InternalLlmMode, LlmProviderId, ModelConfig } from "../../../shared/types";
import type { LlmExecutionRequest, LlmRequestFeatures, LlmRouteDecision } from "./types";

const TOOL_ONLY_PROVIDER: LlmProviderId = "anthropic";

export function getInternalLlmMode(config: Config): InternalLlmMode {
  return config.internalLlm?.mode ?? "prefer-anthropic-with-sdk-fallback";
}

export function extractRequestFeatures(request: Pick<LlmExecutionRequest, "params">): LlmRequestFeatures {
  const tools = request.params.tools ?? [];
  const requiresWebSearch = tools.some((tool) => tool.type === "web_search_20250305");
  const hasJsonCue = request.params.messages.some((message) => {
    if (typeof message.content === "string") {
      return /json/i.test(message.content);
    }
    return message.content.some(
      (block) => block.type === "text" && /json/i.test(block.text),
    );
  });

  return {
    requiresTools: tools.length > 0,
    requiresWebSearch,
    requiresStreaming: false,
    structuredOutputLikely: hasJsonCue,
  };
}

export function getProviderOrder(config: Config, features: LlmRequestFeatures): LlmProviderId[] {
  if (features.requiresTools || features.requiresWebSearch || features.requiresStreaming) {
    return [TOOL_ONLY_PROVIDER];
  }

  const mode = getInternalLlmMode(config);
  switch (mode) {
    case "anthropic-only":
      return ["anthropic"];
    case "sdk-only":
      return ["claude-sdk"];
    case "prefer-sdk-with-anthropic-fallback":
      return ["claude-sdk", "anthropic"];
    case "prefer-anthropic-with-sdk-fallback":
    default:
      return ["anthropic", "claude-sdk"];
  }
}

export function decideRoute(request: LlmExecutionRequest): LlmRouteDecision {
  const attemptedProviderIds = getProviderOrder(request.config, request.features);
  return {
    primaryProviderId: attemptedProviderIds[0] ?? "anthropic",
    fallbackProviderId: attemptedProviderIds[1] ?? null,
    attemptedProviderIds,
  };
}

export function resolveProviderModelId(_feature: keyof ModelConfig, config: Config): string | null {
  void config;
  return null;
}
