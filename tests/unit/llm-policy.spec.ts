import { test, expect } from "@playwright/test";
import type { Config } from "../../src/shared/types";
import { decideRoute, extractRequestFeatures } from "../../src/main/services/llm/provider-policy";

function makeConfig(overrides?: Partial<Config>): Config {
  return {
    maxEmails: 50,
    model: "claude-sonnet-4-20250514",
    dryRun: false,
    analysisPrompt: "analyze",
    draftPrompt: "draft",
    enableSenderLookup: true,
    theme: "system",
    inboxDensity: "compact",
    undoSendDelay: 5,
    keyboardBindings: "superhuman",
    ...overrides,
  };
}

test.describe("llm provider policy", () => {
  test("prefers anthropic then sdk for basic requests by default", () => {
    const params = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 256,
      messages: [{ role: "user" as const, content: "hello" }],
    };

    const decision = decideRoute({
      params,
      options: { caller: "test" },
      config: makeConfig(),
      features: extractRequestFeatures({ params }),
    });

    expect(decision.attemptedProviderIds).toEqual(["anthropic", "claude-sdk"]);
  });

  test("routes tool requests to anthropic only", () => {
    const params = {
      model: "claude-haiku-4-5-20251001",
      max_tokens: 128,
      tools: [{ type: "web_search_20250305" as const, name: "web_search", max_uses: 1 }],
      messages: [{ role: "user" as const, content: "find this person" }],
    };

    const decision = decideRoute({
      params,
      options: { caller: "test" },
      config: makeConfig({ internalLlm: { mode: "prefer-sdk-with-anthropic-fallback" } }),
      features: extractRequestFeatures({ params }),
    });

    expect(decision.attemptedProviderIds).toEqual(["anthropic"]);
  });

  test("respects sdk-only mode for basic requests", () => {
    const params = {
      model: "claude-sonnet-4-20250514",
      max_tokens: 64,
      messages: [{ role: "user" as const, content: "hello" }],
    };

    const decision = decideRoute({
      params,
      options: { caller: "test" },
      config: makeConfig({ internalLlm: { mode: "sdk-only" } }),
      features: extractRequestFeatures({ params }),
    });

    expect(decision.attemptedProviderIds).toEqual(["claude-sdk"]);
  });
});
