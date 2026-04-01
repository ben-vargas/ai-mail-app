import type { Config, InternalLlmReadiness, LlmTaskAvailability } from "../../../shared/types";
import { claudeSdkProvider, resetClaudeSdkProbeCache } from "./providers/claude-sdk-provider";
import { getProviderOrder } from "./provider-policy";

function hasAnthropicApiKey(config: Config): boolean {
  return Boolean(process.env.ANTHROPIC_API_KEY || config.anthropicApiKey);
}

function getTaskAvailability(hasAnthropic: boolean, hasSdk: boolean): LlmTaskAvailability {
  return {
    coreGeneration: hasAnthropic || hasSdk,
    streaming: hasAnthropic,
    toolUse: hasAnthropic,
    webSearch: hasAnthropic,
    usageAccounting: hasAnthropic || hasSdk,
  };
}

export function resetLlmReadinessCaches(): void {
  resetClaudeSdkProbeCache();
}

export async function getInternalLlmReadiness(config: Config): Promise<InternalLlmReadiness> {
  const anthropicAvailable = hasAnthropicApiKey(config);
  const sdkStatus = await claudeSdkProvider.isAvailable(config);
  const providerOrder = getProviderOrder(config, {
    requiresTools: false,
    requiresWebSearch: false,
    requiresStreaming: false,
    structuredOutputLikely: false,
  });

  const taskAvailability = getTaskAvailability(anthropicAvailable, sdkStatus.available);
  const preferredProvider = providerOrder.find((providerId) =>
    providerId === "anthropic" ? anthropicAvailable : sdkStatus.available,
  );
  const fallbackProvider = providerOrder.find(
    (providerId) => providerId !== preferredProvider && (providerId === "anthropic" ? anthropicAvailable : sdkStatus.available),
  );

  return {
    hasAnthropicApiKey: anthropicAvailable,
    hasInternalLlm: taskAvailability.coreGeneration,
    preferredProvider: preferredProvider ?? null,
    fallbackProvider: fallbackProvider ?? null,
    providers: [
      {
        providerId: "anthropic",
        available: anthropicAvailable,
        authMode: anthropicAvailable ? "api_key" : "unknown",
        reason: anthropicAvailable ? undefined : "Anthropic API key not configured",
        capabilities: ["basic-text", "structured-output", "streaming", "tool-use", "web-search", "usage-accounting"],
      },
      sdkStatus,
    ],
    tasks: taskAvailability,
  };
}
