import path from "path";
import { createRequire } from "module";
import { spawn as cpSpawn } from "child_process";
import type { SpawnOptions as SdkSpawnOptions } from "@anthropic-ai/claude-agent-sdk";

const require = createRequire(import.meta.url);

export const resolvedClaudeSdkCliPath = (() => {
  const sdkEntry = require.resolve("@anthropic-ai/claude-agent-sdk");
  const sdkDir = path.dirname(sdkEntry);
  return path.join(sdkDir, "cli.js").replace(/app\.asar([/\\])/, "app.asar.unpacked$1");
})();

export function buildClaudeSdkEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
  );
}

export function spawnClaudeSdkProcess(options: SdkSpawnOptions) {
  return cpSpawn(process.execPath, options.args, {
    cwd: options.cwd,
    env: { ...options.env, ELECTRON_RUN_AS_NODE: "1" },
    stdio: ["pipe", "pipe", "pipe"],
    signal: options.signal,
  });
}
