import type { Message, MessageCreateParamsNonStreaming } from "@anthropic-ai/sdk/resources/messages";
import type { Config } from "../../../shared/types";

export type LlmProviderId = "anthropic" | "claude-sdk";
export type LlmAuthMode = "api_key" | "claude_login" | "unknown";
export type LlmCapability = "basic-text" | "structured-output" | "streaming" | "tool-use" | "web-search" | "usage-accounting";

export interface LlmCallOptions {
  caller: string;
  emailId?: string;
  accountId?: string;
  timeoutMs?: number;
}

export interface LlmRequestFeatures {
  requiresTools: boolean;
  requiresWebSearch: boolean;
  requiresStreaming: boolean;
  structuredOutputLikely: boolean;
}

export interface LlmExecutionRequest {
  params: MessageCreateParamsNonStreaming;
  options: LlmCallOptions;
  config: Config;
  features: LlmRequestFeatures;
}

export interface LlmProviderResult {
  message: Message;
  providerId: LlmProviderId;
  authMode: LlmAuthMode;
  usage: Record<string, number>;
  durationMs: number;
  costCents?: number;
  costEstimated?: boolean;
}

export interface LlmProviderAvailability {
  providerId: LlmProviderId;
  available: boolean;
  authMode: LlmAuthMode;
  reason?: string;
  capabilities: LlmCapability[];
}

export interface LlmProvider {
  readonly id: LlmProviderId;
  readonly capabilities: LlmCapability[];
  isAvailable(config: Config): Promise<LlmProviderAvailability>;
  execute(request: LlmExecutionRequest): Promise<LlmProviderResult>;
}

export interface LlmRouteDecision {
  primaryProviderId: LlmProviderId;
  fallbackProviderId: LlmProviderId | null;
  attemptedProviderIds: LlmProviderId[];
}
