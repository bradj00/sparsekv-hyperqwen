// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
//
// stable-prefix: an opencode plugin that keeps the system prompt byte-identical across agents, so a server with
// prefix caching (vLLM, and the sparse-KV host tier) can share it between them. See docs/AGENT_HARNESS_TIPS.md.
//
// What it does, in the system prompt only (the conversation is untouched):
//   1. moves the per-run <env> block (working directory, date, platform, git status) to the END;
//   2. rewrites absolute paths under the session's working directory to ./relative ones;
//   3. replaces the per-server "You are powered by the model named <served id>. The exact model ID is <id>" line with
//      one neutral line, so agents on servers with different served-model names (same weights) still share the prefix.
//
// Install: copy this file to <opencode config dir>/plugin/stable-prefix.ts, e.g. ~/.config/opencode/plugin/ (global)
// or .opencode/plugin/ (per project). Set STABLE_PREFIX_MODEL to the model name to state, e.g. "Qwen3.8-27B";
// unset, the line names no model.
//
// It uses opencode's experimental "experimental.chat.system.transform" hook; check it still exists after an opencode
// upgrade (the plugin is a no-op if the hook is never called).

const MODEL_LINE = process.env.STABLE_PREFIX_MODEL
  ? `You are powered by the model ${process.env.STABLE_PREFIX_MODEL}.`
  : "You are powered by a large language model."

const ENV_BLOCK = /Here is some useful information about the environment you are running in:\n<env>[\s\S]*?<\/env>\n?/

export const StablePrefix = async ({ directory }: any) => ({
  "experimental.chat.system.transform": async (_input: any, output: any) => {
    output.system.forEach((s: string, k: number) => {
      if (typeof s !== "string") return
      let t = s.replace(/You are powered by the model named [^\n]*/, MODEL_LINE)
      const env = t.match(ENV_BLOCK)
      if (env) t = t.replace(env[0], "")
      if (directory) {
        t = t
          .split("Instructions from: " + directory + "/").join("Instructions from: ")
          .split(directory + "/").join("./")
      }
      output.system[k] = env ? t.trimEnd() + "\n\n" + env[0].trimEnd() : t
    })
  },
})
